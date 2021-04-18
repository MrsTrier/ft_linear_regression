def predict(model=None):
    if model is None:
        try:
            with open('theta_value_file', 'r') as f:
                value = f.readlines()
                index = value[0].index('=')
                theta0 = value[0][index + 1:]
                theta1 = value[1][index + 1:]
        except Exception as e:
            print("Error: please run ft_linear_regression.py before")
            print("{}".format(e))
            exit(0)
        while 1:
            X = input("Enter X to predict Y: ")
            if X == "":
                return
            try:
                X = int(X)
            except Exception as e:
                print("Error: {}".format(e))
                continue
            estimated_Y = float(theta0) + float(theta1) * X
            print(f"Estimated Y is {round(estimated_Y, 3)}")
    else:
        while 1:
            X = input("Enter {} to predict {}: ".format(model.X.name, model.Y.name))
            if X == "":
                return
            try:
                X = int(X)
            except Exception as e:
                print("Error: {}".format(e))
                continue
            estimated_Y = model.theta0 + model.theta1 * X
            print(f"Estimated {model.Y.name} is {round(estimated_Y, 3)}")


if __name__ == '__main__':
    predict()
