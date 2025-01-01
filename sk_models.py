from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def svm_class(X_train, X_test, Y_train, Y_test, param_grid):
    svm_classifer = SVC(decision_function_shape='ovr', random_state=42)
    algo = GridSearchCV(estimator=svm_classifer, param_grid=param_grid, cv=10, scoring='f1_weighted', n_jobs=-1)
    grid_result = algo.fit(X_train, Y_train)
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):  # 输出所有参数组合及得分
        print("%f  with:   %r" % (mean, param))
    best_param = algo.best_params_
    print(best_param)
    # 使用最优参数组合进行训练
    svc = SVC(decision_function_shape='ovr', C=best_param['C'], kernel=best_param['kernel'])
    svc.fit(X_train, Y_train)
    test_result = svc.predict(X_test)
    print('准确率为：', metrics.accuracy_score(Y_test, test_result))
    print(metrics.classification_report(Y_test, test_result))
    print('f1_score is', metrics.f1_score(Y_test, test_result, average='weighted'))
    return best_param


def randomforest_class(X_train, X_test, Y_train, Y_test, param_grid):
    rfc = RandomForestClassifier(random_state=42)
    algo = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, scoring='f1_weighted', n_jobs=-1)
    grid_result = algo.fit(X_train, Y_train)
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):  # 输出所有参数组合及得分
        print("%f  with:   %r" % (mean, param))
    best_param = algo.best_params_
    print(best_param)
    # 使用最优参数组合进行训练
    rfc = RandomForestClassifier(max_depth=best_param['max_depth'], min_samples_leaf=best_param['min_samples_leaf'],
                                 n_jobs=-1)
    rfc.fit(X_train, Y_train)
    test_result = rfc.predict(X_test)
    print('准确率为：', metrics.accuracy_score(Y_test, test_result))
    print(metrics.classification_report(Y_test, test_result))
    print('f1_score is', metrics.f1_score(Y_test, test_result, average='weighted'))
    return best_param


def knn_class(X_train, X_test, Y_train, Y_test, param_grid):
    knn = KNeighborsClassifier()
    algo = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10, scoring='f1_weighted', n_jobs=-1)
    grid_result = algo.fit(X_train, Y_train)
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):  # 输出所有参数组合及得分
        print("%f  with:   %r" % (mean, param))
    best_param = algo.best_params_
    print(best_param)
    # 使用最优参数组合进行训练
    knn = KNeighborsClassifier(n_neighbors=best_param['n_neighbors'], n_jobs=-1)
    knn.fit(X_train, Y_train)
    test_result = knn.predict(X_test)
    print('准确率为：', metrics.accuracy_score(Y_test, test_result))
    print(metrics.classification_report(Y_test, test_result))
    print('f1_score is', metrics.f1_score(Y_test, test_result, average='weighted'))
    return best_param