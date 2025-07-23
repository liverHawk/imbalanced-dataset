import os
import yaml

def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    print(params)
    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    

if __name__ == "__main__":
    main()
