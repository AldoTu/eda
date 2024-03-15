# -*- coding: utf-8 -*-
import pandas.errors

# Import libs
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.base.model
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.formula.api import ols
    from tqdm import tqdm
except ImportError:
    raise ImportError('La librería no se pudo inicializar porque falta una de las dependencias. Ejecuta "pip install -r requirements.txt" para solucionar el problema')

class EDA():

    def __init__(self, verbose: int=0, df: pd.DataFrame=None):
        self.verbose = verbose
        self.df = df

    # Private methods

    def __print_df_stats(self) -> None:
        print("\nImprimiendo valores nulos del dataset\n")
        print("#####################################")
        self.df.info()
        print("\nImprimiendo valores descriptivos del dataset\n")
        print("#####################################")
        print(self.df.describe())
        print("\nImprimiendo primeros registros del dataset para validación\n")
        print("#####################################")
        print(self.df.head(5))

    def __get_dataframe_corr(self) -> pd.DataFrame:
        try:
            return self.df.corr()
        except:
            raise Exception

    def __create_vif_dataframe(self) -> pd.DataFrame:
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.df.columns
        vif_data["VIF"] = [variance_inflation_factor(self.df.values, i) for i in range(len(self.df.columns))]
        vif_data = vif_data.sort_values(by="VIF", ascending=False)
        print(vif_data)
        return vif_data

    # Public methods

    def read_file(self, path: str) -> None:
        try:
            if '.csv' in path:
                self.df = pd.read_csv(path)
                self.__print_df_stats()
            elif '.xlsx' in path:
                self.df = pd.read_excel(path)
                self.__print_df_stats()
            else:
                print('Por favor proporciona la ruta a un archivo .csv o .xlsx')
        except:
            raise Exception

    def convert_to_int(self, column: str) -> None:
        try:
            self.df = self.df[column].astype(int)
        except pandas.errors.IntCastingNaNError:
            print('No se pueden convertir los valores, elimina primero los valores NaN o Inf antes de continuar')

    def delete_columns(self, columns: list) -> None:
        print('\nColumnas eliminadas: ' + str(columns) + "\n")
        self.df.drop(columns=columns, axis=1, inplace=True)
        print(self.df.head(5))

    def analyze_null_values(self):
        total_rows = self.df.shape[0]
        results = {}
        for column in self.df.columns:
            non_null_values = self.df[column].count()
            if non_null_values != total_rows:
                results[column] = non_null_values
        missing_values_df = pd.DataFrame({
            'features': [key for key in results.keys()],
            'non-null values': [values for values in results.values()],
            'non-null %': [((value/total_rows)*100) for value in results.values()]
        })
        print("\nTotal no. of rows: " + str(total_rows) + "\n")
        print(missing_values_df)

    # Functions to plot specific column as a line plot
    def plot_feature(self, column: str) -> None:
        self.df[column].plot()
        plt.show()

    # Functions to plot histograms

    def plot_histograms(self) -> None:
        try:
            for col in tqdm(self.df.columns):
                if self.df[col].dtype != str or self.df[col].dtype != bool:
                    plt.hist(self.df[col], bins=100, ec='black')
                    plt.title(col, fontsize=15)
                    plt.show()
        except:
            raise Exception

    def sns_histograms(self, hue: str="") -> None:
        try:
            f, ax = plt.subplots(figsize=(7, 5))
            sns.set_theme(style="ticks")

            for col in tqdm(self.df.columns):
                sns.histplot(data=self.df, x=col, hue=hue)
                plt.show()
        except:
            raise Exception

    # Function to plot correlation matrix

    def sns_corr_matrix(self) -> None:
        corr = self.__get_dataframe_corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    def sns_pairplot(self, hue: str="") -> None:
        print("Generando gráfico: esta acción puede tardar unos minutos. NO DETENER LA EJECUCION")
        sns.pairplot(self.df, hue=hue)
        plt.show()

    # Function to convert columns in dataset

    def convert_categorical_variable(self, column: str) -> None:
        self.df[column] = self.df[column].map( {
            self.df[column].unique()[i]: i for i in list(range(len(self.df[column].unique())))
        } )

    # Function to calculate VIF

    def calculate_vif(self) -> None:
        try:
            self.__create_vif_dataframe()
        except TypeError:
            print("¡Variable categórica encontrada!")
            print("Intentando convertir valores a cuantitativos")
            for col in tqdm(self.df.columns):
                if self.df[col].dtype == str or self.df[col].dtype == object:
                    self.convert_categorical_variable(col)

            # Restart execution
            try:
                self.__create_vif_dataframe()
            except:
                raise Exception

    # Function to generate linear model

    def generate_linear_model_formula(self, y: str) -> str:
        x_formula = []
        for i in self.df.columns:
            if i != y:
                x_formula.append(i)
        return y + " ~ " + " + ".join(x_formula)

    def generate_linear_model(self, y: str = None, formula: str = None) -> statsmodels.base.model.Model:
        if y is None and formula is None:
            print("Ingresa una etiqueta 'y' o una fórmula")
            return
        elif formula is None:
            model = ols(self.generate_linear_model_formula(y), data=self.df).fit()
        else:
            model = ols(formula, data=self.df).fit()
        print(model.summary())
        return model

    def print_anova_table(self, model: statsmodels.base.model.Model) -> None:
        print(sm.stats.anova_lm(model, typ=2))

    def generate_final_linear_model_formula(self, model: statsmodels.base.model.Model) -> None:
        coefficients = model.params

        final_formula = [f"{coefficients[0]:.2f}"]
        for i, coef in tqdm(enumerate(coefficients[1:], start=1)):
            final_formula.append(f"{coef:.2f} * {self.df.columns[i]}")

        # Print linear regression formula
        print("Fórmula final de regresión: ", " + ".join(final_formula))
