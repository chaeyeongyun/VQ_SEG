def change1(parent):
    parent = parent
    parent[1] = 100
def change2(parent):
    parent[1] = 100
def solution():
    parent = [1, 2, 3, 4, 5]
    change2(parent)
    print(parent)

solution()

def change3(val):
    val = val
    val = 4
def change4(val):
    val = 4
def solution2():
    val  = 1
    change4(val)
    print(val)
solution2()