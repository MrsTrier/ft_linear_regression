
import sys
sys.path.append('/Users/Darya/PycharmProjects/ft_linear_regression')
import ft_linear_regression

def predict(model = None):
    if model == None:
        try:
            with open('theta_value_file', 'r') as f:
                value = f.readlines()
                index = value[0].index('=')
                theta0 = value[0][index + 1:]
                theta1 = value[1][index + 1:]
        except Exception as e:
            print("Error: {}".format(e))
        while 1:
            X = input("Enter X to predict Y: ")
            if X == "":
                return
            try:
                X = int(X)
            except Exception as e:
                print("Error: {}".format(e))
                exit(0)
            estimated_Y = float(theta0) + float(theta1) * X
            print(f"Estimated Y is { round(estimated_Y, 3) }")
    else:
        while 1:
            X = input("Enter {} to predict {}: ".format(model.X.name, model.Y.name))
            if X == "":
                return
            try:
                X = int(X)
            except Exception as e:
                print("message: {}".format(e))
                exit(0)
            estimated_Y = model.theta0 + model.theta1 * X
            print(f"Estimated { model.Y.name } is { round(estimated_Y, 3) }")

class Model:
    X = []
    Y = []
    theta0_path = []
    theta1_path = []
    theta0 = 0
    theta1 = 0
    estimated_Y = []
    max_x: int

    def measure_mae(self):
        sum_of_errors = 0
        for y, estimated_y in zip(self.Y, self.estimated_Y):
            sum_of_errors += abs(y - estimated_y)
        mae = sum_of_errors / len(self.X)
        return mae

    def measure_mse(self):
        sum_of_errors = 0
        for y, estimated_y in zip(self.Y, self.estimated_Y):
            sum_of_errors += pow(y - estimated_y, 2)
        mse = sum_of_errors / len(self.X)
        return mse

    def measure_r_2(self):
        variation_of_estimated_y = 0
        variation_of_y = 0
        for y, estimated_y in zip(self.Y, self.estimated_Y):
            variation_of_estimated_y += pow(estimated_y, 2)
            variation_of_y += pow(y, 2)
        r_2 = variation_of_estimated_y / variation_of_y
        return r_2

    def get_fit_quality(self):
        mae = self.measure_mae()
        print(f'MAE: { round(mae, 3) }')
        mse = self.measure_mse()
        print(f'MSE: { round(mse, 3) }')
        r_2 = self.measure_r_2()
        print(f'R2: { round(r_2, 3) }')

    def train(self):
        ft_linear_regression.train(model)

    def predict(self):
        predict(model)

    def print_coefficients(self):
        print(f'Theta1: { round(self.theta1, 3) }')
        print(f'Theta0: { round(self.theta0, 3) }')

if __name__ == '__main__':
    # predict()
    model = Model()
    model.train()
    model.predict()

    ### Раскомментируй чтобы увидеть значния коэффициентов модели ###
    model.print_coefficients()

    ### Раскомментируй чтобы увидеть графики в браузере ###
    ft_linear_regression.plot_data_from(model)
    ft_linear_regression.plot_fitting_process(model)

    ## Раскомментируй чтобы увидеть параметры качества модели ###
    model.get_fit_quality()

