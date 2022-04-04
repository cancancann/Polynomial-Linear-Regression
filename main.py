import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# # Polynomial Linear Regression
# Polynomial Linear Regression Genel Formülü:
# y = a + b1*x + b2*x^2 + b3*x^3 + b4*x^4 + ....... + bN*x^N


df = pd.read_csv("oilCompanySalaries.csv",sep=';')
df.columns


# regression model nesnemizi olan reg nesnemizi oluşturup bunun fit metonu çağırarak x_polynomial ve y eksenlerini fit ediyor
# yani regresyon modelimizi mevcut gerçek verilerle eğitiyoruz:

reg = LinearRegression()
reg.fit(df[['Level']],df['Salary'])

plt.xlabel('Yıl')
plt.ylabel('Maaş')
plt.scatter(df['Level'],df['Salary'])

xekseni = df['Level']
yekseni = reg.predict(df[['Level']])
plt.plot(xekseni,yekseni,color='red',label="Linear Regressions")
plt.legend()
plt.show()

#Linear regression yetersiz gözüküyor.

# bir adet polynomial regression nesnesi oluşturması için PolynomialFeatures fonksiyonunu çağırıyoruz
# Bu fonksiyonu çağırırken polinomun derecesini (N) belirtiyoruz:
polynomial_regression = PolynomialFeatures(degree= 4) #degree ile derecesini belirliyoruz.
x_polynomial = polynomial_regression.fit_transform(df[['Level']])

#Yapay zekayı Eğitme İşlemi
reg = LinearRegression()
reg.fit(x_polynomial,df['Salary'])

y_head = reg.predict(x_polynomial)

plt.plot(df['Level'],y_head, color='green', label='Polynominal Regression')
plt.scatter(df['Level'],df['Salary'],color="red")
plt.xlabel('Level')
plt.ylabel('Salary')
plt.savefig('1.png', dpi= 300)
plt.show()

#Polynomial Regression daha iyi bir Seçim olduğunu gördük.

#Yeni Giren çalışana maaş Belirleme
x_polynomial1 = polynomial_regression.fit_transform([[4.5]])
reg.predict(x_polynomial1)
print(x_polynomial1)
