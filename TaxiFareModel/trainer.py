from sklearn import metrics
from ml_flow_test import EXPERIMENT_NAME
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import joblib
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data



class Trainer():
    def __init__(self, X_train, y_train, exp_model='LinearRegression v1',
                 regressor=LinearRegression(), **kwargs):
        """
            X_train: pandas DataFrame
            y_train: pandas Series
        """
        self.regressor = regressor
        self.model_name = exp_model
        self.pipeline = self.set_pipeline(regressor)
        self.X_train = X_train
        self.y_train = y_train

        # Log to score and params to MLFlow server
        self._mlflow_uri = "https://mlflow.lewagon.co/"
        self.mlflow_local = False
        self.experiment_name = "[DE] [Munich] [kickermeister] TaxiFareModel"
        self.mlflow_log_param("model", exp_model)
        self.mlflow_log_param("user", 'kickermeister')
        for key, value in kwargs.items():
            self.mlflow_log_param(key, value)

    def set_pipeline(self, regressor):
        '''sets a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', regressor)
        ])
        return pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X_train, self.y_train)

    def cross_val_score(self):
        """Return cross-validated score"""
        scores = cross_val_score(self.pipeline,
                                 self.X_train,
                                 self.y_train,
                                 scoring='neg_root_mean_squared_error',
                                 n_jobs=-1)
        score = -1 * scores.mean()
        self.mlflow_log_metric('RMSE', score)
        return score

    def predict(self, X_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return y_pred

    def save_model(self, save_name='model'):
        """ Save the trained model into a .joblib file """
        joblib.dump(self.pipeline, save_name+ '.joblib')

    @memoized_property
    def mlflow_client(self):
        if self.mlflow_local == False:
            mlflow.set_tracking_uri(self._mlflow_uri)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)

    # hold out
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # build pipeline
    trainer = Trainer(X_train, y_train)

    score = trainer.cross_val_score()
    print(f'RMSE of cross-validation: {round(score, 2)}')

    # train the pipeline
    trainer.run()

    # evaluate the pipeline
    #rmse = trainer.predict(X_val)
    #print(f'RMSE of X_val: {round(rmse, 2)}')

    trainer.save_model()
