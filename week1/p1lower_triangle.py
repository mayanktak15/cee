def lower_triangle(n):
    for i in range(1, n + 1):
        print('*' * i)

if __name__ == "__main__":
    n = int(input("Enter number of rows: "))
    lower_triangle(n)
