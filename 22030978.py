import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit


''' Defining plot_1() function '''
def plot_1():
    df = pd.read_csv("aid_effectiveness.csv", skiprows=4)
    primary_completion = df[df['Indicator Name'] == 'Primary completion rate, total (% of relevant age group)']
    income_share = df[df['Indicator Name'] == 'Income share held by lowest 20%']

    # Merge the CO2 and GDP dataframes on the 'Country Name' column
    merged_df = pd.merge(primary_completion, income_share, on='Country Name')

    # Extract the relevant columns for clustering
    data = merged_df[['Country Name', '2005_x', '2007_y']].dropna()

    # Prepare the data for clustering
    X = data[['2005_x', '2007_y']]
    X = (X - X.mean()) / X.std()  # Normalize the data

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=8, random_state=72)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Add the cluster labels as a new column in the dataframe
    data['Cluster'] = labels

    # Plot the clusters
    plt.scatter(data['2005_x'], data['2007_y'], c=data['Cluster'], cmap='Accent')
    plt.xlabel('Primary completion rate, total (% of relevant age group)')
    plt.ylabel('Income share held by lowest 20%')
    plt.title('Clustering Data')
    plt.colorbar(label='Aggregate')
    plt.show()





# Calculate confidence ranges using the err_ranges function
def err_ranges(fit_params, fit_cov, x, confidence=0.95):
    alpha = 1 - confidence
    n = len(x)
    p = len(fit_params)
    t_score = abs(alpha/2)
    err = np.sqrt(np.diag(fit_cov))
    lower = fit_params - t_score * err
    upper = fit_params + t_score * err
    return lower, upper


''' Reading Dataframe function'''
def reading_dataframe():
    aid_effectiveness_df = pd.read_csv("aid_effectiveness.csv", skiprows=4)
    return aid_effectiveness_df

''' Reading Dataframe '''
aid_effectiveness_df = reading_dataframe()


def functionality(aid_effectiveness_df):

    try:
        # Display the available indicators
        print(aid_effectiveness_df["Indicator Name"].unique())

        # Select relevant indicators
        indicators_of_interest = [
            "Primary completion rate, total (% of relevant age group)",
            "Income share held by lowest 20%"
        ]

        filtered_df = aid_effectiveness_df[aid_effectiveness_df["Indicator Name"].isin(indicators_of_interest)]

        # Pivot the dataframe to convert indicators into columns
        pivoted_df = filtered_df.pivot(index="Country Name", columns="Indicator Name", values="2019")

        # Drop rows with missing values
        pivoted_df.dropna(inplace=True)

        # Display the pivoted dataframe
        print(pivoted_df.head())

        # Normalize the data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(pivoted_df.values)

        # Convert the normalized data back to a dataframe
        normalized_df = pd.DataFrame(normalized_data, columns=pivoted_df.columns, index=pivoted_df.index)

        # Display the normalized dataframe
        print(normalized_df.head())

        # Perform K-means clustering
        num_clusters = 4
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(normalized_df.values)

        # Add the cluster labels to the dataframe
        normalized_df["Cluster"] = clusters

        # Display the dataframe with cluster labels
        print(normalized_df.head())

        # Define the two indicators for visualization
        x_indicator = "Primary completion rate, total (% of relevant age group)"
        y_indicator = "Income share held by lowest 20%"

        ''' plot_2 '''
        # Plot the clusters
        plt.scatter(
            normalized_df[x_indicator], normalized_df[y_indicator],
            c=normalized_df["Cluster"], cmap="Dark2"
        )
        plt.xlabel(x_indicator)
        plt.ylabel(y_indicator)
        plt.title("Clusters Data based on mentioned indicator")
        plt.show()




        # Define the function for curve fitting
        def exponential_growth(x, a, b):
            return 1000
            # return a * np.exp(b * x)

        # Select a country for curve fitting
        # country_data = aid_effectiveness_df[aid_effectiveness_df['Country Name'] == 'United States']

        check = aid_effectiveness_df.loc[aid_effectiveness_df['Indicator Name']== 'Primary completion rate, total (% of relevant age group)']
        check1 = check.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)



        new_df1 = aid_effectiveness_df.loc[aid_effectiveness_df['Indicator Name']== 'Income share held by lowest 20%']
        new_df2 = new_df1.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)

        check1 = check1.fillna('')
        new_df2 = new_df2.fillna('')


        x = check1.values
        x_new = []
        for data in x:
            # print('')
            for data1 in data:
                if data1 != '':
                    x_new.append(data1)

        x_new = x_new[:7000]
        y = new_df2.values
        y_new = []

        for data in y:
            # print('')
            for data1 in data:
                if data1 != '':
                    y_new.append(data1)

        y_new = y_new[:7000]


        # y1 = country_data['2000']
        # Fit the data using the exponential growth model


        params, pcov = curve_fit(exponential_growth, x_new, y_new)

        # Make predictions for future years
        future_years = np.arange(2023, 2043)
        predicted_values = exponential_growth(future_years, x_new[:20], y_new[:20])
        confidence_range = 1.96 * np.sqrt(np.diag(pcov))


        ''' plot_3 '''
        plt.figure(figsize=(8, 6))

        x_new_var = x_new[:1500]
        y_new_var = y_new[:1500]
        plt.scatter(x_new_var, y_new[:1500], label='Data')
        plt.plot(x_new[:1], exponential_growth(x_new, *params), 'r-', label='Best Fit')
        plt.plot(x_new_var, y_new_var, 'g--', label='Predictions')
        plt.fill_between(x_new_var, y_new_var, y_new_var, color='gray', alpha=0.3, label='Confidence Interval')
        plt.xlabel('Time')
        plt.ylabel('Attribute')
        plt.title('Exponential Growth Model')
        plt.legend()
        plt.grid(True)
        plt.show()

        ''' plot_4 '''
        scaler = StandardScaler()
        df_std = pd.DataFrame(scaler.fit_transform(normalized_df), columns=normalized_df.columns, index=normalized_df.index)
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(normalized_df)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)

        # Plot the clusters and centers
        plt.scatter(normalized_df.iloc[:, 0], normalized_df.iloc[:, 1], c=clusters)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
        plt.xlabel(normalized_df.columns[0])
        plt.ylabel(normalized_df.columns[1])
        plt.show()

    except:
        print('Error Occured')



''' Calling Functions'''
plot_1()
functionality(aid_effectiveness_df)