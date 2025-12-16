def main():
    print("Hello from agenticcybersense!")

    def dummy_function():
        print("This is a dummy function.")

    def dummy_function_with_params(param1, param2):
        return f"Received parameters: {param1} and {param2}"

    result = dummy_function_with_params("value1", "value2")
    print(result)


if __name__ == "__main__":
    main()
