import pandas as pd
from scipy.stats import f_oneway, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt


class HypothesisTesting:
    def __init__(self, data):
        self.data = data

    def test_claims_across_provinces(self):
        province_groups = [self.data[self.data['Province'] == p]['Total_Claim'] for p in self.data['Province'].unique()]
        result = f_oneway(*province_groups)
        return {
            "Test": "ANOVA",
            "Null Hypothesis": "No significant claims differences across provinces",
            "F-Statistic": result.statistic,
            "p-Value": result.pvalue,
            "Reject Null": result.pvalue < 0.05
        }

    def test_claims_difference_gender(self):
        male_claims = self.data[self.data['Gender'] == 'Male']['Total_Claim']
        female_claims = self.data[self.data['Gender'] == 'Female']['Total_Claim']
        result = ttest_ind(male_claims, female_claims, equal_var=False)
        return {
            "Test": "T-Test",
            "Null Hypothesis": "No significant claims differences between genders",
            "T-Statistic": result.statistic,
            "p-Value": result.pvalue,
            "Reject Null": result.pvalue < 0.05
        }

    def visualize_distributions(self):
        sns.boxplot(x='Province', y='Total_Claim', data=self.data)
        plt.title('Claims Distribution by Province')
        plt.show()

        sns.boxplot(x='Gender', y='Total_Claim', data=self.data)
        plt.title('Claims Distribution by Gender')
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv("... data/insurance_data.csv") #data/preprocessed_data.csv")
    tester = HypothesisTesting(data)

    print(tester.test_claims_across_provinces())
    print(tester.test_claims_difference_gender())
    tester.visualize_distributions()
