def pyramid(n):
    for i in range(1, n + 1):
        spaces = ' ' * (n - i)
        stars = '*' * (2 * i - 1)
        print(spaces + stars)

if __name__ == "__main__":
    n = int(input("Enter number of rows: "))
    pyramid(n)
