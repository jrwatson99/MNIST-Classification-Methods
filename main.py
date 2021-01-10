from KNN import knn
from feed_forward import feed_forward
from CNN import cnn
from ecnn import ensemble_cnn


def knn_menu():
    knn.open_menu()


def ff_menu():
    feed_forward.open_menu()


def cnn_menu():
    cnn.open_menu()


def ecnn_menu():
    ensemble_cnn.open_menu()


menu_switcher = {
    'knn': knn_menu,
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
                       '(ff) Feed Forward Neural Network\n'
                       '(cnn) Convolutional Neural Network\n'
                       '(ecnn) Ensemble Convolutional Neural Network\n'
                       '(q) Quit\n')
        print(menuMessage)
        menu_option = input("Select an option: ")

        menu_func = menu_switcher.get(menu_option, "nothing")
        menu_func()

            # case 'knn':
            #
            # if menu_option == 'v':
            #     view_menu(x_test, y_test)
            # elif menu_option == 't':
            #     classify_test_menu(kd_tree, y_train, x_test)
            # #elif menu_option == 'c':
            # #    print("")
            # elif menu_option == 'a':
            #     calc_accuracy(kd_tree, y_train, x_test, y_test)