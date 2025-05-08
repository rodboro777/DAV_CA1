import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import scipy.cluster.hierarchy as sch
import geopandas as gpd
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from scipy import stats
from sklearn.cluster import KMeans


class DataAnalyzer:
    def __init__(self):
        self.data_path = "YALE-EPI.xlsx"
        self.df = None
        self.scaler = StandardScaler()
        self.pca = None
        self.components = None
        self.var_ratio = None

    def load_data(self):
        """Load and preprocess YALE-EPI data"""
        try:
            print("Loading environmental performance data")
            raw_df = pd.read_excel(self.data_path)
            last_10_years = [str(year) for year in range(2013, 2023)]
            yearly_dfs = []

            for year in last_10_years:
                year_df = raw_df.pivot(
                    index="Economy Name", columns="Indicator", values=year
                )
                yearly_dfs.append(year_df)

            self.df = pd.concat(yearly_dfs).groupby(level=0).mean()
            self.df.columns = self.df.columns.str.replace(
                "Environmental Performance Index: ", ""
            )

            indicators = [
                "PM2.5 exposure",
                "SO2 exposure",
                "Household solid fuels",
                "Unsafe drinking water",
                "Unsafe sanitation",
                "Lead exposure",
                "Biodiversity Habitat Index",
                "Terrestrial biome protection (global weights)",
                "Tree cover loss",
                "Ocean Plastics",
                "Greenhouse gas emissions per capita",
                "Ozone exposure",
                "Recycling",
            ]

            self.df = self.df[indicators]
            self.df = self.df.interpolate(method="linear", axis=0)
            self.df = self.df.fillna(self.df.median())

            # Handle zero values with median imputation
            zero_mask = self.df == 0
            for column in self.df.columns:
                column_zero_mask = zero_mask[column]
                if column_zero_mask.any():
                    self.df.loc[column_zero_mask, column] = self.df.loc[
                        ~column_zero_mask, column
                    ].median()

            self.df.to_excel("EPI_Preprocessed.xlsx")
            print("Data preprocessing completed.")

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("Running correlation analysis")
        correlation_matrix = self.df.corr()
        p_values = self.df.corr(method=lambda x, y: stats.pearsonr(x, y)[1])

        plt.figure(figsize=(12, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            annot_kws={"size": 8},
        )
        plt.title("Correlation Matrix of Environmental Performance Indicators")
        plt.tight_layout()
        plt.savefig("results/correlation/correlation_matrix.png")
        plt.close()

        correlation_matrix.to_csv("results/correlation/correlation_matrix.csv")
        p_values.to_csv("results/correlation/p_values.csv")

        health_indicators = [
            "PM2.5 exposure",
            "Unsafe drinking water",
            "Unsafe sanitation",
        ]
        climate_indicators = [
            "Greenhouse gas emissions per capita",
            "Ocean Plastics",
            "Recycling",
        ]

        plt.figure(figsize=(15, 10))

        # Only plot significant correlations (p < 0.05)
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if p_values.iloc[i, j] < 0.05:
                    x_col = correlation_matrix.columns[i]
                    y_col = correlation_matrix.columns[j]
                    x = self.df[x_col]
                    y = self.df[y_col]

                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                    plt.figure(figsize=(8, 6))
                    plt.scatter(x, y, alpha=0.5)
                    plt.plot(
                        x,
                        intercept + slope * x,
                        "r",
                        label=f"r={r_value:.2f}, p={p_value:.2e}",
                    )
                    plt.title(f"Correlation Analysis: {x_col} vs {y_col}")
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.legend()
                    plt.tight_layout()

                    plot_filename = (
                        f"results/correlation/{x_col}_vs_{y_col}.png".replace(" ", "_")
                    )
                    plt.savefig(plot_filename)
                    plt.close()

        return correlation_matrix, p_values

    def run_pca(self, n_components=None):
        """Scale data for PCA"""
        print("Performing PCA")
        scaled_data = self.scaler.fit_transform(self.df)
        self.pca = PCA(n_components=n_components)
        self.components = self.pca.fit_transform(scaled_data)
        self.var_ratio = self.pca.explained_variance_ratio_

        variance_df = pd.DataFrame(
            {
                "PC": range(1, len(self.var_ratio) + 1),
                "Explained_Variance": self.var_ratio,
                "Cumulative_Variance": np.cumsum(self.var_ratio),
            }
        )
        variance_df.to_csv("results/pca/variance_explained.csv", index=False)

        """
        # Scree plot
        plt.figure(figsize=(10, 6))
        plt.bar(
            variance_df["PC"],
            variance_df["Explained_Variance"],
            alpha=0.5,
            align="center",
            label="Individual explained variance"
        )
        plt.plot(
            variance_df["PC"],
            variance_df["Cumulative_Variance"],
            "r-",
            marker="o",
            label="Cumulative explained variance"
        )
        plt.ylabel("Explained variance ratio")
        plt.xlabel("Principal components")
        plt.xticks(variance_df["PC"])
        plt.legend(loc="best")
        plt.tight_layout()
        plt.title("Scree Plot of Principal Components")
        plt.savefig("results/pca/explained_variance.png")
        plt.close()

        # Component scatter plots
        pc_combinations = [(0, 1), (0, 2), (1, 2)]
        for pc_i, pc_j in pc_combinations:
            if self.components.shape[1] <= max(pc_i, pc_j):
                continue
            
            plt.figure(figsize=(12, 8))
            plt.scatter(
                self.components[:, pc_i],
                self.components[:, pc_j],
                alpha=0.6
            )
            plt.xlabel(f"PC{pc_i+1}")
            plt.ylabel(f"PC{pc_j+1}")
            plt.title(f"Principal Component Analysis: PC{pc_i+1} vs PC{pc_j+1}")
            plt.tight_layout()
            plt.savefig(f"results/pca/pca_scatter_{pc_i+1}_{pc_j+1}.png")
            plt.close()
        """

        return self.components

    def feature_importance(self):
        """Get feature importance from PCA"""
        feature_importance = pd.DataFrame(
            {"Feature": self.df.columns, "Importance": np.abs(self.pca.components_[0])}
        ).sort_values("Importance", ascending=False)

        # Save feature importance
        feature_importance.to_csv("results/pca/feature_importance.csv", index=False)

    def hierarchical_cluster(self, max_clusters=10):
        """Perform Hierarchical Clustering Analysis"""
        cluster_data = self.components[:, :7]
        distances = pdist(cluster_data, metric="euclidean")
        Z = linkage(distances, method="ward")

        linkage_methods = ["ward", "single", "complete", "average"]
        for method in linkage_methods:
            plt.figure(figsize=(15, 8))
            dendrogram = sch.dendrogram(
                sch.linkage(cluster_data, method=method),
                labels=self.df.index,
                leaf_rotation=90,
            )
            plt.title(f"Hierarchical Clustering Dendrogram using {method} Linkage")
            plt.xlabel("Countries")
            plt.ylabel("Distance")
            plt.tight_layout()
            plt.savefig(f"results/clustering/dendrogram_{method}.png")
            plt.close()

        # Performs hierarchical clustering with optimal number of clusters
        optimal_k = 3
        agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
        cluster_labels = agg_clustering.fit_predict(cluster_data)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            self.components[:, 0],
            self.components[:, 1],
            c=cluster_labels,
            cmap="viridis",
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Hierarchical Clustering Results on Principal Components")
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig("results/clustering/hierarchical_clusters.png")
        plt.close()

        return cluster_labels

    def kmeans_cluster(self, max_clusters=10):
        """Perform K-means Clustering"""
        cluster_data = self.components[:, :7]
        inertias = []
        k_values = range(1, max_clusters + 1)
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(cluster_data)
            inertias.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, "bx-")
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method for Optimal Number of Clusters")
        plt.tight_layout()
        plt.savefig("results/clustering/kmeans_elbow.png")
        plt.close()

        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_data)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.components[:, 0],
            self.components[:, 1],
            c=cluster_labels,
            cmap="viridis",
            alpha=0.6,
        )

        # Label outlier countries on the plot
        pc1_scores = self.components[:, 0]
        pc2_scores = self.components[:, 1]
        top_pc1 = np.argsort(np.abs(pc1_scores))[-5:]
        top_pc2 = np.argsort(np.abs(pc2_scores))[-5:]
        countries_to_label = np.unique(np.concatenate([top_pc1, top_pc2]))

        for idx in countries_to_label:
            country = self.df.index[idx]
            plt.annotate(
                country,
                (self.components[idx, 0], self.components[idx, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("K-means Clustering Results on Principal Components")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(
            "results/clustering/kmeans_clusters.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f"PC{i+1}" for i in range(len(self.pca.components_))],
            index=self.df.columns,
        )

        return cluster_labels, loadings

    def dbscan_cluster(self, eps=1.0, min_samples=3):
        """Perform DBSCAN Clustering (for comparison)"""
        cluster_data = self.components[:, :7]
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(cluster_data)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            self.components[:, 0],
            self.components[:, 1],
            c=cluster_labels,
            cmap="viridis",
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("DBSCAN Clustering Results on Principal Components")
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig("results/clustering/dbscan_clusters.png")
        plt.close()

        return cluster_labels

    def save_results(self, output_path):
        """Save PCA results to a file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pc_df = pd.DataFrame(
            self.components,
            columns=[f"PC{i+1}" for i in range(self.components.shape[1])],
        )
        pc_df.to_csv(output_path, index=False)

        explained_variance_df = pd.DataFrame(
            {
                "Component": [f"PC{i+1}" for i in range(len(self.var_ratio))],
                "Explained_Variance": self.var_ratio,
                "Cumulative_Variance": np.cumsum(self.var_ratio),
            }
        )
        explained_variance_df.to_csv("results/pca/explained_variance.csv", index=False)

    def composite_score(self, weight_method="pca"):
        print(f"Calculating composite scores using {weight_method} weights")
        scaled_data = self.scaler.fit_transform(self.df)

        sub_indices = {
            "Health": [
                "PM2.5 exposure",
                "SO2 exposure",
                "Household solid fuels",
                "Unsafe drinking water",
                "Unsafe sanitation",
                "Lead exposure",
                "Ozone exposure",
            ],
            "Ecosystems": [
                "Biodiversity Habitat Index",
                "Terrestrial biome protection (global weights)",
                "Tree cover loss",
            ],
            "Waste": ["Ocean Plastics", "Recycling"],
            "Climate": ["Greenhouse gas emissions per capita"],
        }

        if weight_method == "pca":
            feature_importance = pd.read_csv("results/pca/feature_importance.csv")
            weights = (
                feature_importance.set_index("Feature")["Importance"]
                .reindex(self.df.columns)
                .values
            )
            weights /= weights.sum()
        else:
            weights = np.ones(len(self.df.columns)) / len(self.df.columns)

        # Normalize weights within each sub-index
        sub_index_scores = pd.DataFrame(index=self.df.index)
        for sub_index, cols in sub_indices.items():
            valid_cols = [col for col in cols if col in self.df.columns]
            if not valid_cols:
                continue

            sub_weights = weights[[self.df.columns.get_loc(col) for col in valid_cols]]
            sub_weights /= sub_weights.sum()
            sub_index_scores[sub_index] = np.sum(
                scaled_data[:, [self.df.columns.get_loc(col) for col in valid_cols]]
                * sub_weights,
                axis=1,
            )

        self.scores = sub_index_scores.mean(axis=1)
        composite_df = pd.DataFrame(
            {"Country": self.df.index, "Composite_Score": self.scores}
        ).merge(sub_index_scores, left_index=True, right_index=True)

        Path("results/composite").mkdir(parents=True, exist_ok=True)
        Path("results/visualization").mkdir(parents=True, exist_ok=True)

        composite_df.to_csv("results/composite/composite_indicator.csv")

        composite_df["Rank"] = (
            composite_df["Composite_Score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        composite_df.sort_values("Rank", inplace=True)

        # Save top and bottom 10 countries
        top_10 = composite_df.head(10)
        bottom_10 = composite_df.tail(10)

        top_10.to_csv("results/composite/top_10_countries.csv", index=False)
        bottom_10.to_csv("results/composite/bottom_10_countries.csv", index=False)

        # Save ranked data
        composite_df.to_csv(
            "results/composite/composite_indicator_ranked.csv", index=False
        )

        try:
            composite_df = composite_df.reset_index()
            composite_df["Country"] = composite_df["Country"].str.strip()
            # Fix country name mappings for visualization
            composite_df["Country"] = composite_df["Country"].replace(
                {
                    "Korea, Rep.": "South Korea",
                    "Sao Tome and Principe": "São Tomé and Príncipe",
                }
            )

            fig = px.choropleth(
                composite_df,
                locations="Country",
                locationmode="country names",
                color="Composite_Score",
                hover_name="Country",
                color_continuous_scale="Viridis",
                title="Global Environmental Performance Composite Score Map",
                template="plotly_white",
            )

            fig.update_layout(
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor="Black",
                    showland=True,
                    landcolor="LightGray",
                    showcountries=True,
                    countrycolor="Black",
                    projection_type="natural earth",
                ),
                margin=dict(l=0, r=0, t=50, b=0),
            )

            fig.write_html("results/visualization/composite_world_map.html")
            fig.write_image("results/visualization/composite_world_map.png", scale=2)
        except Exception as e:
            print(f"Warning: Could not create world map visualization: {str(e)}")

        print(
            f"Top 3 performing countries:\n{composite_df.head(3)[['Country', 'Composite_Score']]}\n"
        )
        print(
            f"Bottom 3 performing countries:\n{composite_df.tail(3)[['Country', 'Composite_Score']]}"
        )

        return composite_df


if __name__ == "__main__":
    analyzer = DataAnalyzer()
    analyzer.load_data()
    Path("results/correlation").mkdir(parents=True, exist_ok=True)

    correlation_matrix, p_values = analyzer.correlation_analysis()
    analyzer.run_pca(n_components=7)
    analyzer.feature_importance()
    analyzer.save_results("results/pca/pca_results.csv")

    cluster_labels = analyzer.hierarchical_cluster()
    kmeans_labels, loadings = analyzer.kmeans_cluster(max_clusters=10)
    dbscan_labels = analyzer.dbscan_cluster()
    composite_df = analyzer.composite_score(weight_method="pca")
