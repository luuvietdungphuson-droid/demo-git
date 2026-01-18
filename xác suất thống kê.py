import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, shapiro, kstest

print("Đang tải dữ liệu Titanic...")
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

print("✓ Dữ liệu đã được tải thành công!")
print(f"Số dòng: {len(df)}")
print(f"Các cột: {list(df.columns)}")

print("\n" + "=" * 80)
print("1. XỬ LÝ MISSING VALUES")
print("=" * 80)
print("\nSố lượng missing values theo cột:")
print(df.isnull().sum())

df_clean = df.copy()
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
df_clean['Cabin'].fillna('Unknown', inplace=True)
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)

print("\nSố lượng missing values sau khi xử lý:")
print(df_clean.isnull().sum())

print("\n" + "=" * 80)
print("2. PHÁT HIỆN VÀ XỬ LÝ OUTLIERS (IQR)")
print("=" * 80)

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

outliers_age, lower_age, upper_age = detect_outliers_iqr(df_clean, 'Age')
print(f"\nAge - Outliers:")
print(f"Lower bound: {lower_age:.2f}")
print(f"Upper bound: {upper_age:.2f}")
print(f"Số lượng outliers: {len(outliers_age)}")

outliers_fare, lower_fare, upper_fare = detect_outliers_iqr(df_clean, 'Fare')
print(f"\nFare - Outliers:")
print(f"Lower bound: {lower_fare:.2f}")
print(f"Upper bound: {upper_fare:.2f}")
print(f"Số lượng outliers: {len(outliers_fare)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].boxplot(df_clean['Age'].dropna())
axes[0].set_title('Boxplot of Age', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Age')
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(df_clean['Fare'].dropna())
axes[1].set_title('Boxplot of Fare', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Fare')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("3. KIỂM TRA PHÂN PHỐI CHUẨN")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_clean['Age'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title('Histogram of Age', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

axes[1].hist(df_clean['Fare'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[1].set_title('Histogram of Fare', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fare')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
plt.show()

shapiro_age = shapiro(df_clean['Age'].dropna())
shapiro_fare = shapiro(df_clean['Fare'].dropna())

print(f"\nShapiro-Wilk Test:")
print(f"Age: statistic={shapiro_age[0]:.4f}, p-value={shapiro_age[1]:.4e}")
print(f"Fare: statistic={shapiro_fare[0]:.4f}, p-value={shapiro_fare[1]:.4e}")

ks_age = kstest(df_clean['Age'].dropna(), 'norm', args=(df_clean['Age'].mean(), df_clean['Age'].std()))
ks_fare = kstest(df_clean['Fare'].dropna(), 'norm', args=(df_clean['Fare'].mean(), df_clean['Fare'].std()))

print(f"\nKolmogorov-Smirnov Test:")
print(f"Age: statistic={ks_age[0]:.4f}, p-value={ks_age[1]:.4e}")
print(f"Fare: statistic={ks_fare[0]:.4f}, p-value={ks_fare[1]:.4e}")

print("\n" + "=" * 80)
print("4. PHÂN TÍCH THỐNG KÊ")
print("=" * 80)

print("\nThống kê mô tả:")
print(df_clean[['Age', 'Fare']].describe())

def confidence_interval_95(data):
    mean = np.mean(data)
    std_error = stats.sem(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=std_error)
    return mean, ci

mean_age, ci_age = confidence_interval_95(df_clean['Age'].dropna())
mean_fare, ci_fare = confidence_interval_95(df_clean['Fare'].dropna())

print(f"\n95% Confidence Intervals:")
print(f"Age: Mean={mean_age:.2f}, CI=({ci_age[0]:.2f}, {ci_age[1]:.2f})")
print(f"Fare: Mean={mean_fare:.2f}, CI=({ci_fare[0]:.2f}, {ci_fare[1]:.2f})")

print("\n" + "=" * 80)
print("5. KIỂM ĐỊNH GIẢ THIẾT")
print("=" * 80)

survived_age = df_clean[df_clean['Survived'] == 1]['Age'].dropna()
not_survived_age = df_clean[df_clean['Survived'] == 0]['Age'].dropna()

t_stat, t_pvalue = ttest_ind(survived_age, not_survived_age)

print(f"\nT-test (Age vs Survival):")
print(f"Mean Age (Survived): {survived_age.mean():.2f}")
print(f"Mean Age (Not Survived): {not_survived_age.mean():.2f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {t_pvalue:.4e}")

if t_pvalue < 0.05:
    print("Kết luận: Có sự khác biệt có ý nghĩa thống kê về tuổi giữa người sống sót và không sống sót")
else:
    print("Kết luận: Không có sự khác biệt có ý nghĩa thống kê về tuổi giữa người sống sót và không sống sót")

contingency_table = pd.crosstab(df_clean['Pclass'], df_clean['Survived'])
chi2, chi_pvalue, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square Test (Pclass vs Survival):")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {chi_pvalue:.4e}")
print(f"Degrees of freedom: {dof}")

if chi_pvalue < 0.05:
    print("Kết luận: Có mối liên hệ có ý nghĩa thống kê giữa hạng vé và tỷ lệ sống sót")
else:
    print("Kết luận: Không có mối liên hệ có ý nghĩa thống kê giữa hạng vé và tỷ lệ sống sót")

pclass_groups = [df_clean[df_clean['Pclass'] == i]['Fare'].dropna() for i in [1, 2, 3]]
f_stat, anova_pvalue = stats.f_oneway(*pclass_groups)

print(f"\nANOVA (Fare vs Pclass):")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {anova_pvalue:.4e}")

if anova_pvalue < 0.05:
    print("Kết luận: Có sự khác biệt có ý nghĩa thống kê về giá vé giữa các hạng")
else:
    print("Kết luận: Không có sự khác biệt có ý nghĩa thống kê về giá vé giữa các hạng")

print("\n" + "=" * 80)
print("6. HỒI QUY VÀ TƯƠNG QUAN")
print("=" * 80)

correlation_matrix = df_clean[['Age', 'Fare', 'Survived', 'Pclass']].corr()
print("\nMa trận tương quan:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.3f')
plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

from scipy.stats import linregress

age_fare_clean = df_clean[['Age', 'Fare']].dropna()
slope, intercept, r_value, p_value, std_err = linregress(age_fare_clean['Age'], age_fare_clean['Fare'])

print(f"\nHồi quy tuyến tính (Age vs Fare):")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4e}")

plt.figure(figsize=(10, 6))
plt.scatter(age_fare_clean['Age'], age_fare_clean['Fare'], alpha=0.5)
plt.plot(age_fare_clean['Age'], slope * age_fare_clean['Age'] + intercept, 'r', linewidth=2)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Fare', fontsize=12)
plt.title('Linear Regression: Age vs Fare', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_age_fare.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("7. INSIGHTS CHÍNH")
print("=" * 80)

print(f"\n1. Tỷ lệ sống sót tổng thể: {df_clean['Survived'].mean()*100:.2f}%")

print(f"\n2. Tỷ lệ sống sót theo giới tính:")
print(df_clean.groupby('Sex')['Survived'].agg(['mean', 'count']))

print(f"\n3. Tỷ lệ sống sót theo hạng vé:")
print(df_clean.groupby('Pclass')['Survived'].agg(['mean', 'count']))

print(f"\n4. Tỷ lệ sống sót theo nhóm tuổi:")
df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                labels=['Child', 'Teen', 'Adult', 'Middle Age', 'Senior'])
print(df_clean.groupby('Age_Group')['Survived'].agg(['mean', 'count']))

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

survival_sex = df_clean.groupby('Sex')['Survived'].mean()
axes[0, 0].bar(survival_sex.index, survival_sex.values, color=['skyblue', 'lightcoral'])
axes[0, 0].set_title('Survival Rate by Sex', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Survival Rate')
axes[0, 0].grid(True, alpha=0.3, axis='y')

survival_pclass = df_clean.groupby('Pclass')['Survived'].mean()
axes[0, 1].bar(survival_pclass.index.astype(str), survival_pclass.values, color=['gold', 'silver', 'brown'])
axes[0, 1].set_title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Survival Rate')
axes[0, 1].set_xlabel('Pclass')
axes[0, 1].grid(True, alpha=0.3, axis='y')

survival_age_group = df_clean.groupby('Age_Group')['Survived'].mean()
axes[1, 0].bar(range(len(survival_age_group)), survival_age_group.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
axes[1, 0].set_xticks(range(len(survival_age_group)))
axes[1, 0].set_xticklabels(survival_age_group.index, rotation=45)
axes[1, 0].set_title('Survival Rate by Age Group', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Survival Rate')
axes[1, 0].grid(True, alpha=0.3, axis='y')

df_clean.boxplot(column='Fare', by='Pclass', ax=axes[1, 1])
axes[1, 1].set_title('Fare Distribution by Passenger Class', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Pclass')
axes[1, 1].set_ylabel('Fare')
plt.suptitle('')

plt.tight_layout()
plt.savefig('insights_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("KẾT LUẬN CHÍNH")
print("=" * 80)
print("\n1. Tuổi: Hành khách trẻ tuổi có tỷ lệ sống sót cao hơn")
print("2. Hạng vé: Hành khách hạng nhất có tỷ lệ sống sót cao nhất")
print("3. Giới tính: Nữ giới có tỷ lệ sống sót cao hơn nam giới đáng kể")
print("4. Giá vé: Có sự bất bình đẳng lớn về giá vé giữa các hành khách")
print("5. Yếu tố kinh tế xã hội (Pclass, Fare) đóng vai trò quan trọng trong tỷ lệ sống sót")
print("\n" + "=" * 80)