from methods import knn, linear_svc, svm, feed_forward, cnn, ensemble_cnn


def knn_menu():
    knn.open_menu()


def linear_svc_menu():
    linear_svc.open_menu()


def svm_menu():
    svm.open_menu()


def ff_menu():
    feed_forward.open_menu()


def cnn_menu():
    cnn.open_menu()


def ecnn_menu():
    ensemble_cnn.open_menu()


menu_switcher = {
    'knn': knn_menu,
    'lsvc': linear_svc_menu,
    'svm': svm_menu,
    'ff': ff_menu,
    'cnn': cnn_menu,
    'ecnn': ecnn_menu
}

if __name__ == '__main__':

    #User Menu - options: view test image(int in range), classify test image(int in rage), Test accuracy, classify custom image
    menu_option = 'c'
    while menu_option != 'q':
        menuMessage = ('Main Menu:\n'
                       '(knn) K Nearest Neighbor\n'
                       '(lsvc) Linear Support Vector Classifier\n'
                       '(svm) Support Vector Machine\n'
                       '(ff) Feed Forward Neural Network\n'
                       '(cnn) Convolutional Neural Network\n'
                       '(ecnn) Ensemble Convolutional Neural Network\n'
                       '(q) Quit\n')
        print(menuMessage)
        menu_option = input("Select an option: ")

        if menu_option in menu_switcher:
            menu_func = menu_switcher.get(menu_option, "nothing")
            menu_func()