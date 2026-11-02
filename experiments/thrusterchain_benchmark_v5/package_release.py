"""V4 release packaging is deliberately unavailable in this v5 development copy."""


def package(output):
    raise RuntimeError("The inherited v4 packager is not a v5 release process; see README.md")


if __name__ == "__main__":
    raise SystemExit("V5 release packaging is not yet established; see README.md")
