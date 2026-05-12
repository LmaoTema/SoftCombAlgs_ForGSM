
from core.none_mode import NonePipeline
from core.full_mode import FullPipeline
from core.half_mode import HalfPipeline
from core.pipeline import ProcessingMode


def create_pipeline(
    mode,
    encoder,
    interleaver,
    modulator,
    channel,
    estimator,
    matched_filter,
    equalizer,
    detector,
    deinterleaver,
    decoder,
    soft_llr_generator,
    combiner
):

    common_args = dict(
        encoder=encoder,
        interleaver=interleaver,
        modulator=modulator,
        channel=channel,
        estimator=estimator,
        matched_filter=matched_filter,
        equalizer=equalizer,
        detector=detector,
        deinterleaver=deinterleaver,
        decoder=decoder,
        soft_llr_generator=soft_llr_generator,
        combiner=combiner
    )

    if mode == ProcessingMode.NONE:
        return NonePipeline(**common_args)

    elif mode == ProcessingMode.FULL:
        return FullPipeline(**common_args)

    elif mode == ProcessingMode.HALF:
        return HalfPipeline(**common_args)

    else:
        raise ValueError(f"Unknown processing mode: {mode}")

