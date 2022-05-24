def put(price, strike):
    return max(0, strike-price)


def call(price, strike):
    return max(0, price-strike)


def straddle(price, strike):
    return max(0, strike-price) + max(0, price-strike)


if __name__ == "__main__":
    print(put(10,20))

