import re

def filter_words(input_file, output_file, max_length):
    with open(input_file, 'r') as file:
        words = file.readlines()

    # Regex pattern to match words with only English letters
    pattern = re.compile("^[A-Za-z]+$")

    filtered_words = [word for word in words if len(word.strip()) <= max_length and pattern.match(word.strip())]

    with open(output_file, 'w') as file:
        file.writelines(filtered_words)

def main():
    input_file = 'fruits.txt'  # Change this to your input file name
    output_file = 'fruits_tokens.txt'

    try:
        max_length = int(input("Enter the maximum number of characters for a word: "))
        filter_words(input_file, output_file, max_length)
        print(f"Filtered words have been saved to {output_file}")
    except ValueError:
        print("Please enter a valid number.")
    except FileNotFoundError:
        print(f"The file {input_file} was not found. Please check the file name and try again.")

if __name__ == "__main__":
    main()
