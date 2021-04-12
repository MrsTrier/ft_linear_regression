import pandas as pd
import numpy as np
import plotly.graph_objects as go


def update_theta0(df, learning_rate, theta0):
    theta0 = theta0 - (learning_rate * df.errors.sum() / float(df.shape[0]))
    return theta0


def update_theta1(df, learning_rate, theta1):
    df['error_km'] = df.errors * df.km_scaled
    theta1 = theta1 - (learning_rate * df.error_km.sum() / float(df.shape[0]))
    return theta1


def update_df_with_estimated_price(df, theta0, theta1):
    df['estimated_Y'] = df.km_scaled * theta1 + theta0
    return df


def update_df_with_errors(df):
    df['errors'] = df.estimated_Y - df.Y
    return df

def scale_x(df):
    df['km_scaled'] = df.X / max(df.X)

def write_results_into(model, df):
    model.estimated_Y = df.estimated_Y
    model.max_x = max(df.X)

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
        model.theta1_path.append(model.theta1)
        df = update_df_with_estimated_price(df, model.theta0, model.theta1)
        df = update_df_with_errors(df)
        mse_tmp = mse
        mse = pow(df.errors, 2).sum() / float(df.shape[0])
    write_results_into(model, df)
    return mse

def plot_data_from(model):
    picture = go.Figure()
    picture.add_trace(go.Scatter(x=model.X, y=model.Y, mode='markers', name='sample data'))
    picture.update_layout(go.Layout(xaxis_title=model.X.name, yaxis_title=model.Y.name,
                                    title="{} as a function of {}".format(model.Y.name, model.X.name)))
    picture.add_trace(go.Scatter(x=model.X, y=model.estimated_Y, mode='lines', name='regression line'))
    picture.show()

def plot_fitting_process(model):
    picture = go.Figure()
    path_len = len(model.theta0_path)
    if (path_len > 5):
        idx = np.round(np.linspace(0, path_len - 1, 5)).astype(int)
        for i in idx:
            esimated_Y = model.theta0_path[i] + model.theta1_path[i] * model.X/model.max_x
            picture.add_trace(go.Scatter(x=model.X, y=esimated_Y, mode='lines', name=f'regression line {i}'))
    else:
        i = 0
        for theta0, theta1 in zip(model.theta0_path, model.theta1_path):
            esimated_Y = theta0 + theta1 * model.X/model.max_x
            picture.add_trace(go.Scatter(x=model.X, y=esimated_Y, mode='lines', name=f'regression line {i}'))
            i += 1
    picture.show()



def print_usage():
    print("Usage")

def prepare_df(csv):
    try:
        df = pd.read_csv(csv)
    except FileNotFoundError:
        print("File does not exist")
        exit(0)
    return df

def save_data_to(model, df):
    model.X = df.km
    model.Y = df.price

def rename_columns(df):
    df.columns = ['X', 'Y']
    return df

def train(model):
    csv = input("Enter path to csv (without any braces): ")

    df = prepare_df(csv)
    save_data_to(model, df)
    df = rename_columns(df)
    mse = fit(df, model)