"""Reward transform."""

MODES = ("rebrac_x100", "iql_minus1", "cql_scale_bias", "none")
DEFAULT_MODE = "rebrac_x100"
DEFAULT_SCALE = 10.0
DEFAULT_BIAS = -5.0
HOST_ANTMAZE_MODES = {
    "rebrac": ("rebrac_x100",),
    "iql": ("iql_minus1",),
    "cql": ("cql_scale_bias",),
    "td3_bc": ("iql_minus1",),
    "unifloral": (),
    "corl_iql": ("iql_minus1",),
    "corl_td3_bc": ("iql_minus1",),
    "corl_awac": ("iql_minus1",),
    "corl_cql": ("cql_scale_bias",),
    "corl_rebrac": ("rebrac_x100",),
    "corl_sac_n": ("none",),
    "corl_edac": ("none",),
    "corl_bc": ("none",),
}


def is_antmaze(dataset_name):
    return "antmaze" in dataset_name


def check_reward_transform(
    host, dataset_name, mode, reward_scale=DEFAULT_SCALE, reward_bias=DEFAULT_BIAS
):
    if mode not in MODES:
        raise SystemExit(
            "--reward_transform must be one of %s (got %r)" % (" | ".join(MODES), mode)
        )
    scale_set = reward_scale != DEFAULT_SCALE or reward_bias != DEFAULT_BIAS
    if scale_set and mode != "cql_scale_bias":
        raise SystemExit(
            "--reward_scale / --reward_bias are read ONLY under --reward_transform cql_scale_bias (got mode=%r, scale=%r, bias=%r). Refusing rather than silently ignoring them."
            % (mode, reward_scale, reward_bias)
        )
    if host not in HOST_ANTMAZE_MODES:
        raise SystemExit(
            "unknown reward-transform host token %r (expected one of %s)"
            % (host, " | ".join(sorted(HOST_ANTMAZE_MODES)))
        )
    if not is_antmaze(dataset_name):
        return
    allowed = HOST_ANTMAZE_MODES[host]
    if not allowed:
        raise SystemExit(
            "[B1] host %r has NO registered antmaze reward convention, so it refuses antmaze rather than silently inheriting the ReBRAC x100 default. Register the convention in _reward_transform.HOST_ANTMAZE_MODES with its source before running %s."
            % (host, dataset_name)
        )
    if mode not in allowed:
        raise SystemExit(
            "[B1] --reward_transform %r is not registered for host %r on antmaze (registered: %s). The antmaze reward convention is host specific and is invisible to config_hash, so an unregistered pairing is refused at startup rather than pooled into a cell under someone else's convention."
            % (mode, host, " | ".join(allowed))
        )


def apply_reward_transform(
    dataset,
    dataset_name,
    mode,
    reward_scale=DEFAULT_SCALE,
    reward_bias=DEFAULT_BIAS,
    legacy_print=False,
):
    if not is_antmaze(dataset_name):
        return dataset
    r = dataset["rewards"]
    if mode == "rebrac_x100":
        dataset["rewards"] = r * 100.0
        if legacy_print:
            print(
                f"[antmaze] Scaled rewards by 100x (max={dataset['rewards'].max():.1f})"
            )
    elif mode == "iql_minus1":
        dataset["rewards"] = r - 1.0
        print(
            f"[antmaze] reward_transform=iql_minus1 (r-1) min={dataset['rewards'].min():.1f} max={dataset['rewards'].max():.1f}",
            flush=True,
        )
    elif mode == "cql_scale_bias":
        dataset["rewards"] = r * reward_scale + reward_bias
        print(
            f"[antmaze] reward_transform=cql_scale_bias (r*{reward_scale}+{reward_bias}) min={dataset['rewards'].min():.1f} max={dataset['rewards'].max():.1f}",
            flush=True,
        )
    elif mode == "none":
        print(
            "[antmaze] reward_transform=none (raw 0/1 rewards, no convention applied)",
            flush=True,
        )
    else:
        raise SystemExit(
            "--reward_transform must be one of %s (got %r)" % (" | ".join(MODES), mode)
        )
    return dataset
