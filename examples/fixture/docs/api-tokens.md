# API tokens

API tokens let scripts call the platform without a browser sign-in.

## Creating a token

Open your profile, choose "Create API key" and download the credentials file. Keys expire after thirty days.

## Using a token in Python

Load the credentials file and send the token as a bearer header:

```python
# the credentials file you downloaded from your profile
import json
token = json.load(open("credentials.json"))["api_key"]
headers = {"Authorization": f"Bearer {token}"}
```

Never commit the credentials file to a repository.
