from mlflow.server import get_app_client
from mlflow.server.auth.client import AuthServiceClient
import os
from dotenv import load_dotenv, set_key


def main():
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    old_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    new_password = os.getenv("MLFLOW_TRACKING_NEW_PASSWORD")

    auth_client: AuthServiceClient = get_app_client(
        "basic-auth", tracking_uri=tracking_uri
    )

    print(
        f"New Passowrd: {new_password}\nUsername: {username}\nOld Password: {old_password}"
    )

    try:
        auth_client.update_user_password(username=username, password=new_password)
        print("Password updated successfully")

    except Exception as e:
        print(f"Erro ao atualizar a senha: {e}")
        exit(1)

    os.environ["MLFLOW_TRACKING_PASSWORD"] = new_password
    set_key(".env", "MLFLOW_TRACKING_PASSWORD", new_password)
    print("New password stored at .env")


if __name__ == "__main__":
    main()
