# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
import cpa
from cpa.data import load_dataset_splits
from cpa.model import MLP, GeneralizedSigmoid, NBLoss
from sklearn.metrics import r2_score
from torch.autograd import Variable
from torch.distributions import NegativeBinomial
from torch import nn


class CPA(torch.nn.Module):
    """
    Our main module, the CPA autoencoder
    """

    def __init__(
        self,
        num_genes,
        num_drugs,
        num_covariates,
        device="cuda",
        seed=0,
        patience=5,
        loss_ae="gauss",
        doser_type="mlp",
        decoder_activation="linear",
        hparams="",
    ):
        super(CPA, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_covariates = num_covariates
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(hparams)

        # set models
        self.encoder = MLP(
            [num_genes]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [self.hparams["dim"]]
        )

        self.decoder = MLP(
            [self.hparams["dim"]]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [num_genes * 2],
            last_layer_act=decoder_activation,
        )

        self.adversary_drugs = MLP(
            [self.hparams["dim"]]
            + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
            + [num_drugs]
        )

        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.doser_type = doser_type
        if doser_type == "mlp":
            self.dosers = torch.nn.ModuleList()
            for _ in range(num_drugs):
                self.dosers.append(
                    MLP(
                        [1]
                        + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
                        + [1],
                        batch_norm=False,
                    )
                )
        else:
            self.dosers = GeneralizedSigmoid(num_drugs, self.device, nonlin=doser_type)

        if self.num_covariates == [0]:
            pass
        else:
            assert 0 not in self.num_covariates
            self.adversary_covariates = []
            self.loss_adversary_covariates = []
            self.covariates_embeddings = (
                []
            )  # TODO: Continue with checking that dict assignment is possible via covaraites names and if dict are possible to use in optimisation
            for num_covariate in self.num_covariates:
                self.adversary_covariates.append(
                    MLP(
                        [self.hparams["dim"]]
                        + [self.hparams["adversary_width"]]
                        * self.hparams["adversary_depth"]
                        + [num_covariate]
                    )
                )
                self.loss_adversary_covariates.append(torch.nn.CrossEntropyLoss())
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, self.hparams["dim"])
                )
            self.covariates_embeddings = torch.nn.Sequential(
                *self.covariates_embeddings
            )

        self.drug_embeddings = torch.nn.Embedding(self.num_drugs, self.hparams["dim"])
        # losses
        if self.loss_ae == "nb":
            self.loss_autoencoder = NBLoss()
        elif self.loss_ae == 'gauss':
            self.loss_autoencoder = nn.GaussianNLLLoss()

        self.iteration = 0

        self.to(self.device)

        # optimizers
        has_drugs = self.num_drugs > 0
        has_covariates = self.num_covariates[0] > 0
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder, True)
            + get_params(self.decoder, True)
            + get_params(self.drug_embeddings, has_drugs)
        )
        for emb in self.covariates_embeddings:
            _parameters.extend(get_params(emb, has_covariates))

        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )

        _parameters = get_params(self.adversary_drugs, has_drugs)
        for adv in self.adversary_covariates:
            _parameters.extend(get_params(adv, has_covariates))

        self.optimizer_adversaries = torch.optim.Adam(
            _parameters,
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"],
        )

        if has_drugs:
            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams["dosers_lr"],
                weight_decay=self.hparams["dosers_wd"],
            )

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"]
        )

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams["step_size_lr"]
        )

        if has_drugs:
            self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
                self.optimizer_dosers, step_size=self.hparams["step_size_lr"]
            )

        self.history = {"epoch": [], "stats_epoch": []}

    def set_hparams_(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "dim": 128,
            "dosers_width": 128,
            "dosers_depth": 2,
            "dosers_lr": 4e-3,
            "dosers_wd": 1e-7,
            "autoencoder_width": 128,
            "autoencoder_depth": 3,
            "adversary_width": 64,
            "adversary_depth": 2,
            "reg_adversary": 60,
            "penalty_adversary": 60,
            "autoencoder_lr": 3e-4,
            "adversary_lr": 3e-4,
            "autoencoder_wd": 4e-7,
            "adversary_wd": 4e-7,
            "adversary_steps": 3,
            "batch_size": 256,
            "step_size_lr": 45,
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def move_inputs_(self, genes, drugs, covariates):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if genes.device.type != self.device:
            genes = genes.to(self.device)
            if drugs is not None:
                drugs = drugs.to(self.device)
            if covariates is not None:
                covariates = [cov.to(self.device) for cov in covariates]
        return (genes, drugs, covariates)

    def compute_drug_embeddings_(self, drugs):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.
        """

        if self.doser_type == "mlp":
            doses = []
            for d in range(drugs.size(1)):
                this_drug = drugs[:, d].view(-1, 1)
                doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
            return torch.cat(doses, 1) @ self.drug_embeddings.weight
        else:
            return self.dosers(drugs) @ self.drug_embeddings.weight

    def predict(
        self, 
        genes, 
        drugs, 
        covariates, 
        return_latent_basal=False,
        return_latent_treated=False,
    ):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """

        genes, drugs, covariates = self.move_inputs_(genes, drugs, covariates)

        latent_basal = self.encoder(genes)

        latent_treated = latent_basal

        if self.num_drugs > 0:
            latent_treated = latent_treated + self.compute_drug_embeddings_(drugs)
        if self.num_covariates[0] > 0:
            for i, emb in enumerate(self.covariates_embeddings):
                emb = emb.to(self.device)
                latent_treated = latent_treated + emb(
                    covariates[i].argmax(1)
                )  #argmax because OHE

        gene_reconstructions = self.decoder(latent_treated)
        dim = gene_reconstructions.size(1) // 2
        if self.loss_ae == 'gauss':
            # convert variance estimates to a positive value in [1e-3, \infty)
            gene_means = gene_reconstructions[:, :dim]
            gene_vars = F.softplus(gene_reconstructions[:, dim:]).add(1e-3)
            #gene_vars = gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)
            gene_reconstructions = torch.cat([gene_means, gene_vars], dim=1)

        if self.loss_ae == 'nb':
            gene_mus = F.softplus(gene_reconstructions[:, :dim]).add(1e-3)
            gene_thetas = F.softplus(gene_reconstructions[:, dim:]).add(1e-3)
            #gene_reconstructions[:, :dim] = torch.clamp(gene_reconstructions[:, :dim], min=1e-4, max=1e4)
            #gene_reconstructions[:, dim:] = torch.clamp(gene_reconstructions[:, dim:], min=1e-4, max=1e4)
            gene_reconstructions = torch.cat([gene_mus, gene_thetas], dim=1)

        if return_latent_basal:
            if return_latent_treated:
                return gene_reconstructions, latent_basal, latent_treated
            else:
                return gene_reconstructions, latent_basal
        if return_latent_treated:
            return gene_reconstructions, latent_treated
        return gene_reconstructions

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()
        self.scheduler_adversary.step()
        self.scheduler_dosers.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def update(self, genes, drugs, covariates):
        """
        Update CPA's parameters given a minibatch of genes, drugs, and
        cell types.
        """
        genes, drugs, covariates = self.move_inputs_(genes, drugs, covariates)
        gene_reconstructions, latent_basal = self.predict(
            genes,
            drugs,
            covariates,
            return_latent_basal=True,
        )

        dim = gene_reconstructions.size(1) // 2
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions[:, :dim], genes, gene_reconstructions[:, dim:])
        adversary_drugs_loss = torch.tensor([0.0], device=self.device)
        if self.num_drugs > 0:
            adversary_drugs_predictions = self.adversary_drugs(latent_basal)
            adversary_drugs_loss = self.loss_adversary_drugs(
                adversary_drugs_predictions, drugs.gt(0).float()
            )

        adversary_covariates_loss = torch.tensor(
            [0.0], device=self.device
        )
        if self.num_covariates[0] > 0:
            adversary_covariate_predictions = []
            for i, adv in enumerate(self.adversary_covariates):
                adv = adv.to(self.device)
                adversary_covariate_predictions.append(adv(latent_basal))
                adversary_covariates_loss += self.loss_adversary_covariates[i](
                    adversary_covariate_predictions[-1], covariates[i].argmax(1)
                )

        # two place-holders for when adversary is not executed
        adversary_drugs_penalty = torch.tensor([0.0], device=self.device)
        adversary_covariates_penalty = torch.tensor([0.0], device=self.device)

        if self.iteration % self.hparams["adversary_steps"]:

            def compute_gradients(output, input):
                grads = torch.autograd.grad(output, input, create_graph=True)
                grads = grads[0].pow(2).mean()
                return grads

            if self.num_drugs > 0:
                adversary_drugs_penalty = compute_gradients(
                    adversary_drugs_predictions.sum(), latent_basal
                )

            if self.num_covariates[0] > 0:
                adversary_covariates_penalty = torch.tensor([0.0], device=self.device)
                for pred in adversary_covariate_predictions:
                    adversary_covariates_penalty += compute_gradients(
                        pred.sum(), latent_basal
                    )  # TODO: Adding up tensor sum, is that right?

            self.optimizer_adversaries.zero_grad()
            (
                adversary_drugs_loss
                + self.hparams["penalty_adversary"] * adversary_drugs_penalty
                + adversary_covariates_loss
                + self.hparams["penalty_adversary"] * adversary_covariates_penalty
            ).backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            if self.num_drugs > 0:
                self.optimizer_dosers.zero_grad()
            (
                reconstruction_loss
                - self.hparams["reg_adversary"] * adversary_drugs_loss
                - self.hparams["reg_adversary"] * adversary_covariates_loss
            ).backward()
            self.optimizer_autoencoder.step()
            if self.num_drugs > 0:
                self.optimizer_dosers.step()
        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_covariates": adversary_covariates_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item(),
            "penalty_adv_covariates": adversary_covariates_penalty.item(),
        }

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for CPA
        """

        return self.set_hparams_(self, "")


def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)

def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""NB parameterizations conversion
    Parameters
    ----------
    mu :
        mean of the NB distribution.
    theta :
        inverse overdispersion.
    eps :
        constant used for numerical log stability. (Default value = 1e-6)
    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    assert (mu is None) == (
        theta is None
    ), "If using the mu/theta NB parameterization, both parameters must be specified"
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits

def evaluate_disentanglement(autoencoder, dataset):
    """
    Given a CPA model, this function measures the correlation between
    its latent space and 1) a dataset's drug vectors 2) a datasets covariate
    vectors.

    """
    with torch.no_grad():
        _, latent_basal = autoencoder.predict(
            dataset.genes,
            dataset.drugs,
            dataset.covariates,
            return_latent_basal=True,
        )
    
    mean = latent_basal.mean(dim=0, keepdim=True)
    stddev = latent_basal.std(0, unbiased=False, keepdim=True)
    normalized_basal = (latent_basal - mean) / stddev
    criterion = nn.CrossEntropyLoss()
    pert_scores, cov_scores = 0, []

    def compute_score(labels):
        if len(np.unique(labels)) > 1:
            unique_labels = set(labels)
            label_to_idx = {labels: idx for idx, labels in enumerate(unique_labels)}
            labels_tensor = torch.tensor(
                [label_to_idx[label] for label in labels], dtype=torch.long, device=autoencoder.device
            )
            assert normalized_basal.size(0) == len(labels_tensor)
            #might have to perform a train/test split here
            dataset = torch.utils.data.TensorDataset(normalized_basal, labels_tensor)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

            # 2 non-linear layers of size <input_dimension>
            # followed by a linear layer.
            disentanglement_classifier = MLP(
                [normalized_basal.size(1)]
                + [normalized_basal.size(1) for _ in range(2)]
                + [len(unique_labels)]
            ).to(autoencoder.device)
            optimizer = torch.optim.Adam(disentanglement_classifier.parameters(), lr=1e-2)

            for epoch in range(50):
                for X, y in data_loader:
                    pred = disentanglement_classifier(X)
                    loss = Variable(criterion(pred, y), requires_grad=True)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                pred = disentanglement_classifier(normalized_basal).argmax(dim=1)
                acc = torch.sum(pred == labels_tensor) / len(labels_tensor)
            return acc.item()
        else:
            return 0

    if dataset.perturbation_key is not None:
        pert_scores = compute_score(dataset.drugs_names)
    cov_scores = []
    for cov in list(dataset.covariate_names):
        if len(np.unique(dataset.covariate_names[cov])) == 0:
            cov_scores.append([0])
        else:
            cov_scores.append(compute_score(dataset.covariate_names[cov]))
    return [np.mean(pert_scores), *[np.mean(cov_score) for cov_score in cov_scores]]


def evaluate_r2(autoencoder, dataset, genes_control, min_samples=30):
    """
    Measures different quality metrics about an CPA `autoencoder`, when
    tasked to translate some `genes_control` into each of the drug/covariates
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """

    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
    num, dim = genes_control.size(0), genes_control.size(1)

    for pert_category in np.unique(dataset.pert_categories):
        # pert_category category contains: 'celltype_perturbation_dose' info
        de_idx = np.where(
            dataset.var_names.isin(np.array(dataset.de_genes[pert_category]))
        )[0]

        idx = np.where(dataset.pert_categories == pert_category)[0]

        if len(idx) > min_samples:
            emb_drugs = dataset.drugs[idx][0].view(1, -1).repeat(num, 1).clone()
            emb_covars = [
                covar[idx][0].view(1, -1).repeat(num, 1).clone()
                for covar in dataset.covariates
            ]

            genes_predict = (
                autoencoder.predict(genes_control, emb_drugs, emb_covars).detach().cpu()
            )

            mean_predict = genes_predict[:, :dim]
            var_predict = genes_predict[:, dim:]

            if autoencoder.loss_ae == 'nb':
                counts, logits = _convert_mean_disp_to_counts_logits(
                    torch.clamp(
                        torch.Tensor(mean_predict),
                        min=1e-4,
                        max=1e4,
                    ),
                    torch.clamp(
                        torch.Tensor(var_predict),
                        min=1e-4,
                        max=1e4,
                    )
                )
                dist = NegativeBinomial(
                    total_count=counts,
                    logits=logits
                )
                yp_m = dist.mean.mean(0)
                yp_v = dist.variance.mean(0)
            else:
                # predicted means and variances
                yp_m = mean_predict.mean(0)
                yp_v = var_predict.mean(0)
            # estimate metrics only for reasonably-sized drug/cell-type combos

            y_true = dataset.genes[idx, :].numpy()

            # true means and variances
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)

            mean_score.append(r2_score(yt_m, yp_m))
            var_score.append(r2_score(yt_v, yp_v))

            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
            var_score_de.append(r2_score(yt_v[de_idx], yp_v[de_idx]))

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de, var_score, var_score_de]
    ]


def evaluate(autoencoder, datasets):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distribution (ood) splits.
    """

    autoencoder.eval()
    with torch.no_grad():
        stats_test = evaluate_r2(
            autoencoder, 
            datasets["test"].subset_condition(control=False), 
            (datasets["test"]#.subset_condition(control=True)
                .genes)
        )

        disent_scores = evaluate_disentanglement(autoencoder, datasets["test"])
        stats_disent_pert = disent_scores[0]
        stats_disent_cov = disent_scores[1:]

        evaluation_stats = {
            "training": evaluate_r2(
                autoencoder,
                datasets["training"].subset_condition(control=False),
                (datasets["training"]#.subset_condition(control=True)
                    .genes)
            ),
            "test": stats_test,
            "ood": evaluate_r2(
                autoencoder,
                datasets["ood"],
                (datasets["test"]#.subset_condition(control=True)
                    .genes)
            ),
            "perturbation disentanglement": stats_disent_pert,
            "optimal for perturbations": 1 / datasets["test"].num_drugs
            if datasets["test"].num_drugs > 0
            else None,
        }
        if len(stats_disent_cov) > 0:
            for i in range(len(stats_disent_cov)):
                evaluation_stats[
                    f"{list(datasets['test'].covariate_names)[i]} disentanglement"
                ] = stats_disent_cov[i]
                evaluation_stats[
                    f"optimal for {list(datasets['test'].covariate_names)[i]}"
                ] = 1 / datasets["test"].num_covariates[i]
    autoencoder.train()
    return evaluation_stats

def prepare_cpa(args, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_dataset_splits(
        args["data"],
        args["perturbation_key"],
        args["dose_key"],
        args["covariate_keys"],
        args["split_key"],
        args["control"],
    )

    autoencoder = CPA(
        datasets["training"].num_genes,
        datasets["training"].num_drugs,
        datasets["training"].num_covariates,
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        doser_type=args["doser_type"],
        patience=args["patience"],
        hparams=args["hparams"],
        decoder_activation=args["decoder_activation"],
    )
    if state_dict is not None:
        autoencoder.load_state_dict(state_dict)

    return autoencoder, datasets


def train_cpa(args, return_model=False):
    """
    Trains a CPA autoencoder
    """

    autoencoder, datasets = prepare_cpa(args)

    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets["training"],
                batch_size=autoencoder.hparams["batch_size"],
                shuffle=True,
            )
        }
    )

    pjson({"training_args": args})
    pjson({"autoencoder_params": autoencoder.hparams})
    args["hparams"] = autoencoder.hparams

    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        epoch_training_stats = defaultdict(float)

        for data in datasets["loader_tr"]:
            genes, drugs, covariates = data[0], data[1], data[2:]

            minibatch_training_stats = autoencoder.update(genes, drugs, covariates)

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in autoencoder.history.keys()):
                autoencoder.history[key] = []
            autoencoder.history[key].append(epoch_training_stats[key])
        autoencoder.history["epoch"].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        autoencoder.history["elapsed_time_min"] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: patience ran out OR
        # time ran out OR max epochs achieved
        stop = ellapsed_minutes > args["max_minutes"] or (
            epoch == args["max_epochs"] - 1
        )

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            evaluation_stats = evaluate(autoencoder, datasets)
            for key, val in evaluation_stats.items():
                if not (key in autoencoder.history.keys()):
                    autoencoder.history[key] = []
                autoencoder.history[key].append(val)
            autoencoder.history["stats_epoch"].append(epoch)

            pjson(
                {
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                }
            )

            torch.save(
                (autoencoder.state_dict(), args, autoencoder.history),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                ),
            )

            pjson(
                {
                    "model_saved": "model_seed={}_epoch={}.pt\n".format(
                        args["seed"], epoch
                    )
                }
            )
            stop = stop or autoencoder.early_stopping(np.mean(evaluation_stats["test"]))
            if stop:
                pjson({"early_stop": epoch})
                break

    if return_model:
        return autoencoder, datasets


def train_cpa_api(args, state_dict=None):
    adata = sc.read(args['data'])
    if state_dict is None:
        cpa_api = cpa.api.API(
            adata, 
            perturbation_key='condition',
            doser_type='logsigm',
            split_key='split',
            covariate_keys=[],
            only_parameters=False,
            hparams={}, 
        )
    else:
        cpa_api = cpa.api.API(
            adata, 
            pretrained=state_dict
        )

    cpa_api.train(
        max_epochs=args['max_epochs'], 
        run_eval=True, 
        checkpoint_freq=args['checkpoint_freq'],
        filename=None, 
        max_minutes=args['max_minutes']
    )


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser(description="Drug combinations.")
    # dataset arguments
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--perturbation_key", type=str, default="condition")
    parser.add_argument("--control", type=str, default=None)
    parser.add_argument("--dose_key", type=str, default="dose_val")
    parser.add_argument("--covariate_keys", nargs="*", type=str, default=["cell_type"])
    parser.add_argument("--split_key", type=str, default="split")
    parser.add_argument("--loss_ae", type=str, default="gauss")
    parser.add_argument("--doser_type", type=str, default="sigm")
    parser.add_argument("--decoder_activation", type=str, default="linear")

    # CPA arguments (see set_hparams_() in cpa.model.CPA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hparams", type=str, default="")

    # training arguments
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--max_minutes", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--checkpoint_freq", type=int, default=20)

    # output folder
    parser.add_argument("--save_dir", type=str, required=True)
    # number of trials when executing cpa.sweep
    parser.add_argument("--sweep_seeds", type=int, default=200)
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    train_cpa(parse_arguments())
