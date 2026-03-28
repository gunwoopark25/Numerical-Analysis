# Homework #1

# 행렬 곱 C = A * B
# A 크기: m x n
# B 크기: n x p
# 결과 C 크기: m x p


def matmul(A, B):


    m = len(A)        # A의 행 개수
    n = len(A[0])     # A의 열 개수
    p = len(B[0])     # B의 열 개수


    # C = []
    # Python에서 [] 는 "빈 리스트(empty list)"를 의미함
    # 데이터 타입은 list
    # 이후 C는 "리스트 안에 리스트(list of lists)" 구조로 확장되어
    # m x p 행렬을 저장하는 자료구조로 사용됨
    C = []
    

    # 코드가 작동하게끔, 본 함수를 완성하시오.
    
    for i in range(m):
        row = []
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            row.append(s)
        C.append(row)

    return C


# 예제 행렬
A = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]


B = [
    [7.0, 8.0],
    [9.0, 10.0],
    [11.0, 12.0]
]


C = matmul(A, B)


print(C)