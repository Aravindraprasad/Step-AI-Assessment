def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Successfully connected to Milvus")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False

if connect_to_milvus():
    print("Milvus is ready to use")
else:
    print("Failed to connect to Milvus")