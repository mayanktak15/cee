def upper_triangle(n):
    for i in range(n, 0, -1):
        print(' ' * (n - i) + '*' * i)

if __name__ == "__main__":
    n = int(input("Enter number of rows: "))
    upper_triangle(n)
