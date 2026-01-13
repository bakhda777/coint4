from coint2.cli import main

# Allow running `python -m coint2 ...` directly.
# Example:
#     python -m coint2 run --config configs/main_2024.yaml
# Click requires the entry-point function name to be passed explicitly when
# invoking programmatically. If no additional arguments are supplied here, it
# simply forwards control to the CLI which parses sys.argv as usual.

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
