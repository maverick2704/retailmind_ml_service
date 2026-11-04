import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["tenure_bucket"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 24, 48, 60, np.inf],
            labels=["0-12", "13-24", "25-48", "49-60", "60+"]
        ).astype(str)

        X["charge_ratio"] = (
            X["TotalCharges"] / (X["MonthlyCharges"] + 1)
        )

        X["monthly_per_tenure"] = (
            X["MonthlyCharges"] / (X["tenure"] + 1)
        )

        X["is_month_to_month"] = (
            X["Contract"] == "Month-to-month"
        ).astype(int)

        X["is_two_year_contract"] = (
            X["Contract"] == "Two year"
        ).astype(int)

        X["has_online_security"] = (
            X["OnlineSecurity"] == "Yes"
        ).astype(int)

        X["has_tech_support"] = (
            X["TechSupport"] == "Yes"
        ).astype(int)

        X["uses_fiber"] = (
            X["InternetService"] == "Fiber optic"
        ).astype(int)

        X["electronic_check"] = (
            X["PaymentMethod"] == "Electronic check"
        ).astype(int)

        return X