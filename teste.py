valor_inicial = 0
valor_maximo = 1000 

while valor_inicial < valor_maximo:
    print("vc ainda tem credito")
    creditos = int(input("Digite o acrescimo: "))
    print(f"credito {creditos}")

    valor_inicial = valor_inicial + creditos
    print(f"valor_inicial {valor_inicial}")
    print("--------------------")

print("vc n tem mais credito")
