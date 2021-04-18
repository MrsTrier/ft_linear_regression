import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

    def print_coefficients(self):
        print(f'Theta1: { round(self.theta1, 3) }')
        print(f'Theta0: { round(self.theta0, 3) }')


def update_theta0(df, learning_rate, theta0):
    theta0 = theta0 - (learning_rate * df.errors.sum() / float(df.shape[0]))
    return theta0


def update_theta1(df, learning_rate, theta1):
    df['error_mul_by_X'] = df.errors * df.X_scaled
    theta1 = theta1 - (learning_rate * df.error_mul_by_X.sum() / float(df.shape[0]))
    return theta1

def update_df_with_estimated_price(df, theta0, theta1):
    df['estimated_Y'] = df.X_scaled * theta1 + theta0
    return df

def update_df_with_errors(df):
    df['errors'] = df.estimated_Y - df.Y
    return df

def scale_x(df):
    df['X_scaled'] = df.X / max(df.X)

def write_results_into(model, df):
    model.estimated_Y = df.estimated_Y
    model.max_x = max(df.X)

def write_into_file(theta0, theta1):
    try:
        theta_value_file = open('theta_value_file', 'w')
        theta_value_file.write('theta0={}\ntheta1={}'.format(theta0, theta1))
        theta_value_file.close()
    except Exception:
        print("Error: something went wrong while writing into file.")

def fit(df, model):
    learning_rate = 0.1
    scale_x(df)
    mse = pow(df.Y, 2).sum() / float(df.shape[0])
    df = update_df_with_estimated_price(df, model.theta0, model.theta1)
    df = update_df_with_errors(df)
    mse_tmp = mse + 2
    model.theta0_path = [model.theta0]
    model.theta1_path = [model.theta1]
    while mse_tmp - mse > 0.1:
        model.theta0 = update_theta0(df, learning_rate, model.theta0)
        model.theta1 = update_theta1(df, learning_rate, model.theta1)
        model.theta0_path.append(model.theta0)
        model.theta1_path.append(model.theta1/max(df.X))
        df = update_df_with_estimated_price(df, model.theta0, model.theta1)
        df = update_df_with_errors(df)
        mse_tmp = mse
        mse = pow(df.errors, 2).sum() / float(df.shape[0])
    write_into_file(model.theta0, model.theta1/max(df.X))
    write_results_into(model, df)
    model.theta1 = model.theta1/max(df.X)
    return mse

def plot_data_from(model):
    title = '{} as a function of {}'.format(model.Y.name, model.X.name)
    picture = go.Figure()
    picture.add_trace(go.Scatter(x=model.X, y=model.Y, mode='markers', name='sample data'))
    picture.update_layout(xaxis_title=model.X.name, yaxis_title=model.Y.name, title=title)
    picture.add_trace(go.Scatter(x=model.X, y=model.estimated_Y, mode='lines', name='regression line'))
    picture.show()

def plot_fitting_process(model):
    title = 'Fitting process'
    picture = go.Figure()
    path_len = len(model.theta0_path)
    if path_len > 5:
        idx = np.round(np.linspace(0, path_len - 1, 5)).astype(int)
        for i in idx:
            esimated_Y = model.theta0_path[i] + model.theta1_path[i] * model.X
            picture.add_trace(go.Scatter(x=model.X, y=esimated_Y, mode='lines', name=f'regression line {i}'))
    else:
        i = 0
        for theta0, theta1 in zip(model.theta0_path, model.theta1_path):
            esimated_Y = theta0 + theta1 * model.X
            picture.add_trace(go.Scatter(x=model.X, y=esimated_Y, mode='lines', name=f'regression line {i}'))
            i += 1
    picture.show()

def prepare_df(csv):
    try:
        df = pd.read_csv(csv)
    except FileNotFoundError:
        print("Error: file does not exist.")
        exit(0)
    except Exception:
        print("Error: something went wrong. Try another file.")
        exit(0)
    return df

def save_data_to(model, df):
    model.X = df.km
    model.Y = df.price

def rename_columns(df):
    df.columns = ['X', 'Y']
    return df

def train(model = None):
    if (model == None):
        model = Model()
    csv = input("Enter path to csv (without any braces): ")
    df = prepare_df(csv)
    save_data_to(model, df)
    df = rename_columns(df)
    fit(df, model)

if __name__ == '__main__':
    train()