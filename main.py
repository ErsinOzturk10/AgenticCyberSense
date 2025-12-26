"""This is the main module for the agenticcybersense package."""


def main() -> None:
    """Define the main function of the agenticcybersense package."""
    print("Hello from agenticcybersense!")  # noqa: T201

    def dummy_function() -> None:
        print("This is a dummy function.")  # noqa: T201

    dummy_function()

    print("Goodbye from agenticcybersense!")  # noqa: T201
    # test comment,

    def dummy_function_with_params(param1: str, param2: str) -> str:
        """Take parameters and return a combined string.

        Args:
            param1: First parameter.
            param2: Second parameter.

        Returns:
            Combined string of parameters.

        """
        return f"Received parameters: {param1} and {param2}"

    result = dummy_function_with_params("value1", "value2")
    print(result)  # noqa: T201


if __name__ == "__main__":
    main()
