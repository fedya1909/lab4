import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True


# Представим, что предупреждения нас не касаются
warnings.filterwarnings("ignore")


# Загружаем данные
df = pd.read_csv("optdigits.tra")
print(df.info())
print()
print(df.describe())
print()
print(df.head(10))
print()

y_column = df.columns[64]
y = df[y_column]
x = df.drop(columns=[y_column])

df1 = pd.read_csv('optdigits.tes')
print(df1.info())
y1_column = df.columns[64]
y1 = df[y1_column]
x1 = df.drop(columns=[y1_column])


def division_df_val_and_test(data, val_percent=0.2):
    l = round(len(data) * val_percent)
    return data[:l], data[l:]


x_train, x_val = division_df_val_and_test(x1.values)
y_train, y_val = division_df_val_and_test(y1.values)

p = Pipeline([
    ('scaler', StandardScaler()),  # Масштабируем (стандартизируем) признаки
    # Применяем метод главных компонент для сокращения размерности пространства признаков
    ('pca', PCA())
]).fit(x)
plt.plot(np.cumsum(p['pca'].explained_variance_ratio_))
plt.title("Объясненная дисперсия в зависимости от количества признаков")
plt.show()

#Для сохранения 99% информации достаточно 49 признаков

# Будем вычислять score для разного объема выборки


def learning_curve(model):
  scores = []
  sizes = []
  for p in np.linspace(0.1, 1, 5):
    l = round(len(x) * p)
    x_part = x_train[:l]
    y_part = y_train[:l]

    model.fit(x_part, y_part)

    learning_score = model.score(x_part, y_part)
    validation_score = model.score(x_val, y_val)

    scores.append([learning_score, validation_score])
    sizes.append(l)

  return {'size': sizes, 'score': np.array(scores)}


pca_features = 49

mlp_results = dict()
for optimizer in 'lbfgs', 'sgd', 'adam':  # Пробуем разные функции оптимизации
  for lr in np.linspace(0.001, 0.1, 3):  # Коэффициенты обучения
    for alpha in np.linspace(0.0001, 0.1, 3):  # Коэффициенты регуляризации
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_features)),
            ("mlp", MLPClassifier(
                (100,),
                solver=optimizer,
                learning_rate_init=lr,
                alpha=alpha,
                max_iter=300
            )
            )])
        mlp_results[(optimizer, lr, alpha)] = learning_curve(model)

s = sorted(mlp_results.items(), key=lambda x: np.mean(
    x[1]['score'], axis=0)[1])
best, worst = s[-1], s[0]
print(best)
print()
print(worst)
# Выводим лучшую и худшую модели
# В массиве score в первом столбце производительность модели на обучающей выборке, во втором - на валидационной

plot_obj = plt.plot(best[1]['size'], best[1]['score'])
plt.legend(iter(plot_obj), ['Обучающая выборка', 'Валидационная выборка'])
plt.xlabel('Объем выборки')
plt.ylabel('score')
plt.show()

# Изобразим поверхность параметров
stats = dict()
for k, v in mlp_results.items():
  optimizer, learning_rate, regularization_alpha = k
  score = np.mean(v['score'], axis=0)[1]
  stats[optimizer] = stats.get(optimizer, [])
  stats[optimizer].append([learning_rate, regularization_alpha, score])

stats = np.array(stats[best[0][0]])
ax = plt.axes(projection='3d')
ax.plot_trisurf(stats[:, 0], stats[:, 1], stats[:, 2])
ax.set_xlabel("learning rate")
ax.set_ylabel("alpha")
ax.set_zlabel("score")
plt.show()

# Представляем, что нас устраивают параметры и пробуем еще разные конфигурации слоев
# (Еще представляем что этим параметры будут хорошими и для этих конфигураций)
best_solver, best_lr, best_alpha = best[0]
mlp_layers_results = dict()
for first_layer in np.linspace(100, 300, 3):
  for second_layer in np.linspace(20, 80, 3):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_features)),
        ("mlp", MLPClassifier(
            (round(first_layer), round(second_layer)),
            solver=best_solver,
            learning_rate_init=best_lr,
            alpha=best_alpha,
            max_iter=300
        )
        )])
    mlp_layers_results[(first_layer, second_layer)] = learning_curve(model)

s = sorted(mlp_layers_results.items(),
           key=lambda x: np.mean(x[1]['score'], axis=0)[1])
best, worst = s[-1], s[0]
print(best)
print()
print(worst)
# Score не сильно увеличился по сравнению с предыдущим экспериментом, а еще больше нейронов слишком долго обучать

first_layer, second_layer = best[0]
model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=pca_features)),
    ("mlp", MLPClassifier(
        (round(first_layer), round(second_layer)),
        solver=best_solver,
        learning_rate_init=best_lr,
        alpha=best_alpha,
        max_iter=300
    )
    )])
# Наконец проверим score на тестовой выборке
model.fit(x_train, y_train)
print(model.score(x, y))

# Тоже самое для перцептрона
p_results = dict()
for penalty in 'l2', 'l1', 'elasticnet':
  for lr in np.linspace(0.001, 0.1, 3):
    for alpha in np.linspace(0.0001, 0.1, 3):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_features)),
            ("perceptron", Perceptron(
                penalty=penalty,
                eta0=lr,
                alpha=alpha
            )
            )])
        p_results[(penalty, lr, alpha)] = learning_curve(model)

s = sorted(p_results.items(), key=lambda x: np.mean(x[1]['score'], axis=0)[1])
best_p, worst_p = s[-1], s[0]
print(best_p)
print()
print(worst_p)

plot_obj = plt.plot(best_p[1]['size'], best_p[1]['score'])
plt.legend(iter(plot_obj), ['Обучающая выборка', 'Валидационная выборка'])
plt.xlabel('Объем выборки')
plt.ylabel('score')
plt.show()

stats = dict()
for k, v in p_results.items():
  optimizer, learning_rate, regularization_alpha = k
  score = np.mean(v['score'], axis=0)[1]
  stats[optimizer] = stats.get(optimizer, [])
  stats[optimizer].append([learning_rate, regularization_alpha, score])

stats = np.array(stats[best_p[0][0]])
ax = plt.axes(projection='3d')
ax.plot_trisurf(stats[:, 0], stats[:, 1], stats[:, 2])
ax.set_xlabel("learning rate")
ax.set_ylabel("alpha")
ax.set_zlabel("score")
plt.show()

best_penalty_p, best_lr_p, best_alpha_p = best_p[0]
model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=pca_features)),
    ("perceptron", Perceptron(
        penalty=best_penalty_p,
        eta0=best_lr_p,
        alpha=best_alpha_p
    )
    )])
model.fit(x_train, y_train)
print(model.score(x, y))
