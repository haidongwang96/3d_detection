import threading


def task(n):
    for i in range(n):
        print(i)

thread1 = threading.Thread(target=task,args=(100,))
thread2 = threading.Thread(target=task,args=(202,))

thread1.start()
thread2.start()

thread1.join()
thread2.join()

print("finish !")


