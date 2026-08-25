# Portfolio closure and license map

## Released artifacts

The public research artifact is the automatic-gated Hugging Face adapter
[`steven0226/tw-med-llm-qlora-adapter`](https://huggingface.co/steven0226/tw-med-llm-qlora-adapter),
frozen at revision `b1d8f74291da75d0719b5a3ea0d088ee8236e096`. The selected
checkpoint is **step 700**. Repository source and distributable Python packages
are recorded by the GitHub `v0.2.0` release; adapter weights are not bundled in
those code distributions.

`model_card/README.md` is a publication template. Its
`{{HF_ADAPTER_REPO_ID}}` marker is replaced by the release pipeline rather than
being a broken public link. Review the [live rendered model
card](https://huggingface.co/steven0226/tw-med-llm-qlora-adapter) for the public
artifact and use the frozen revision above when comparing it with local
evidence.

## Evaluation evidence

The retained evidence uses the fixed MedQA and TMMLU+ multiple-choice research
protocol described in the repository. Training reached step 703, but step 703
validation was non-finite, so it was rejected and step 700 was selected. The
reported adapter-versus-base delta includes the effect of base-model answer
parse failures; it must not be interpreted entirely as increased medical
knowledge. The task is multiple-choice research, not clinical use, and the
frozen metrics do not establish performance on open-ended care or real patient
records.

## License chain

**MIT covers repository code only**. The **adapter, TAIDE/Gemma base, and datasets retain their own upstream terms**. In particular, adapter use remains
subject to the applicable TAIDE/Gemma license and gated access terms, the base
model requires its own access, and each dataset keeps its source license or
data-card restrictions. Neither the repository MIT license nor this closure
relicenses model weights or research data.

## Safety boundary

This adapter is for research and education and **does not constitute medical advice**, diagnosis, treatment guidance, or a clinical safety system. Outputs
may be wrong, outdated, biased, or incomplete. Health decisions require a
qualified medical professional using the full clinical context; the public
artifact must not be used to automate patient-specific decisions.

## Reproduction boundary

Reproduction means following the pinned code, dataset revisions, base-model
revision, prompt/parser contract, and recorded seed described by the repository.
Access to the gated adapter does not grant access to the gated base model.
This closeout does not retrain, rebuild private artifacts, execute notebooks,
or rerun paid/cloud evaluations. The notebook builders may only be checked for
tracked-source consistency as part of the local release gate.
