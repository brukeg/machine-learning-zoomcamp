## Homework 10 - solutions
```bash
@brukeg ➜ /workspaces/machine-learning-zoomcamp (main) $ kind --version
kind version 0.20.0
@brukeg ➜ /workspaces/machine-learning-zoomcamp (main) $ kubectl version
Client Version: v1.33.2
Kustomize Version: v5.6.0
Server Version: v1.27.3
WARNING: version difference between client (1.33) and server (1.27) exceeds the supported minor version skew of +/-1
@brukeg ➜ /workspaces/machine-learning-zoomcamp (main) $ ls -lh
total 124K
drwxrwxrwx+ 3 codespace codespace 4.0K Sep 28 23:31 01-intro
drwxrwxrwx+ 3 codespace codespace 4.0K Oct  9 05:55 02-regression
drwxrwxrwx+ 3 codespace codespace 4.0K Oct 18 18:05 03-classification
drwxrwxrwx+ 3 codespace codespace 4.0K Oct 20 05:08 04-Evaluation
drwxrwxrwx+ 4 codespace codespace 4.0K Oct 27 00:14 05-deployment
drwxrwxrwx+ 3 codespace codespace 4.0K Dec  3 06:18 06-trees
drwxrwxrwx+ 4 codespace codespace 4.0K Dec  3 06:18 08-deep-learning
drwxrwxrwx+ 7 codespace codespace 4.0K Dec  8 05:11 09-serverless
drwxrwxrwx+ 3 codespace codespace 4.0K Dec 13 04:01 10-kubernetes
-rw-rw-rw-  1 codespace codespace  334 Dec 10 05:56 Pipfile
-rw-r--r--  1 codespace codespace  74K Dec 10 05:57 Pipfile.lock
-rw-rw-rw-  1 codespace root        69 Sep 27 03:23 README.md
drwxrwxrwx+ 2 codespace codespace 4.0K Dec 13 20:16 homework-10
@brukeg ➜ /workspaces/machine-learning-zoomcamp (main) $ git clone https://github.com/DataTalksClub/machine-learning-zoomcamp.git
Cloning into 'machine-learning-zoomcamp'...
remote: Enumerating objects: 6652, done.
remote: Counting objects: 100% (2927/2927), done.
remote: Compressing objects: 100% (579/579), done.
remote: Total 6652 (delta 2440), reused 2348 (delta 2348), pack-reused 3725 (from 2)
Receiving objects: 100% (6652/6652), 13.74 MiB | 24.30 MiB/s, done.
Resolving deltas: 100% (4227/4227), done.
@brukeg ➜ /workspaces/machine-learning-zoomcamp (main) $ cd ./machine-learning-zoomcamp/cohorts/2025/05-deployment/homework/
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ docker build -f Dockerfile_full -t zoomcamp-model:3.13.10-hw10 .
[+] Building 21.2s (16/16) FINISHED                                                                                                                                           docker:default
 => [internal] load build definition from Dockerfile_full                                                                                                                               0.1s
 => => transferring dockerfile: 397B                                                                                                                                                    0.0s
 => [internal] load metadata for ghcr.io/astral-sh/uv:latest                                                                                                                            0.9s
 => [internal] load metadata for docker.io/library/python:3.13.10-slim-bookworm                                                                                                         1.3s
 => [auth] library/python:pull token for registry-1.docker.io                                                                                                                           0.0s
 => [auth] astral-sh/uv:pull token for ghcr.io                                                                                                                                          0.0s
 => [internal] load .dockerignore                                                                                                                                                       0.0s
 => => transferring context: 2B                                                                                                                                                         0.0s
 => FROM ghcr.io/astral-sh/uv:latest@sha256:5cb6b54d2bc3fe2eb9a8483db958a0b9eebf9edff68adedb369df8e7b98711a2                                                                            4.2s
 => => resolve ghcr.io/astral-sh/uv:latest@sha256:5cb6b54d2bc3fe2eb9a8483db958a0b9eebf9edff68adedb369df8e7b98711a2                                                                      0.0s
 => => sha256:5cb6b54d2bc3fe2eb9a8483db958a0b9eebf9edff68adedb369df8e7b98711a2 2.19kB / 2.19kB                                                                                          0.0s
 => => sha256:e64e0ddf4bd05ffaca0b3c35c80971b848d2c733b4979267747567ca2f2a2cb0 669B / 669B                                                                                              0.0s
 => => sha256:4948a7ffcc5566709109f9ec3a8b6bd66472bacddf82c3d487603be4266325ef 1.30kB / 1.30kB                                                                                          0.0s
 => => sha256:6fc8456bf7d7fcb2f47a79e2b32a804a845bdfbaef8c69ff94468c37523b5405 22.39MB / 22.39MB                                                                                        1.1s
 => => sha256:0b2b3b65b0f0ad09c49dac35ec5b71ceb99254d0729e6896adedf66f0420c759 98B / 98B                                                                                                0.2s
 => => extracting sha256:6fc8456bf7d7fcb2f47a79e2b32a804a845bdfbaef8c69ff94468c37523b5405                                                                                               0.8s
 => => extracting sha256:0b2b3b65b0f0ad09c49dac35ec5b71ceb99254d0729e6896adedf66f0420c759                                                                                               0.0s
 => [internal] load build context                                                                                                                                                       0.0s
 => => transferring context: 57.48kB                                                                                                                                                    0.0s
 => [stage-0 1/7] FROM docker.io/library/python:3.13.10-slim-bookworm@sha256:d3fef00fbd9ab948d206fe74c1bdd8105f535e5bd90a6074558c31c328ae48b7                                           7.2s
 => => resolve docker.io/library/python:3.13.10-slim-bookworm@sha256:d3fef00fbd9ab948d206fe74c1bdd8105f535e5bd90a6074558c31c328ae48b7                                                   0.0s
 => => sha256:d3fef00fbd9ab948d206fe74c1bdd8105f535e5bd90a6074558c31c328ae48b7 9.13kB / 9.13kB                                                                                          0.0s
 => => sha256:12fc14bddfdbf774231210f1ec826cd85c0cbfaf1a5615ecf2521320d8df4057 1.75kB / 1.75kB                                                                                          0.0s
 => => sha256:b6cd69eec6b21bdf94e569b18ffbbd2ee934f0aa37b060037803f36bf6fb1419 5.54kB / 5.54kB                                                                                          0.0s
 => => sha256:8e44f01296e3a6fdc31a671bee1c2259c5d5ee8b49f29aec42b5d2af15600296 28.23MB / 28.23MB                                                                                        1.2s
 => => sha256:1531b2cd2260a21320d84edaadce399f6d16c2d79d24cedc8fb82f0c5397f43b 3.52MB / 3.52MB                                                                                          1.1s
 => => sha256:4af252baaf636418207dc490d433d45841662d1f2d09554852a2c5a52c69c9f4 251B / 251B                                                                                              1.3s
 => => extracting sha256:8e44f01296e3a6fdc31a671bee1c2259c5d5ee8b49f29aec42b5d2af15600296                                                                                               3.5s
 => => sha256:cb6fc5a445286cf221641ad90df28d8997c7e7cef6cab302e3ea12c2ed01fb70 12.47MB / 12.47MB                                                                                        1.7s
 => => extracting sha256:1531b2cd2260a21320d84edaadce399f6d16c2d79d24cedc8fb82f0c5397f43b                                                                                               0.2s
 => => extracting sha256:cb6fc5a445286cf221641ad90df28d8997c7e7cef6cab302e3ea12c2ed01fb70                                                                                               0.9s
 => => extracting sha256:4af252baaf636418207dc490d433d45841662d1f2d09554852a2c5a52c69c9f4                                                                                               0.0s
 => [stage-0 2/7] COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/                                                                                                                0.9s
 => [stage-0 3/7] WORKDIR /code                                                                                                                                                         0.1s
 => [stage-0 4/7] COPY pyproject.toml uv.lock .python-version ./                                                                                                                        0.6s
 => [stage-0 5/7] RUN uv sync --locked                                                                                                                                                  7.0s
 => [stage-0 6/7] COPY pipeline_v2.bin .                                                                                                                                                0.0s
 => [stage-0 7/7] COPY q6_predict.py ./                                                                                                                                                 0.0s
 => exporting to image                                                                                                                                                                  3.8s
 => => exporting layers                                                                                                                                                                 3.7s
 => => writing image sha256:8a3cee9bc8ba52289b5b01c6cca8f3bd23f8171a1f02cdc48e21fec252c4881a                                                                                            0.0s
 => => naming to docker.io/library/zoomcamp-model:3.13.10-hw10                                                                                                                          0.0s
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ docker run -it --rm -p 9696:9696 zoomcamp-model:3.13.10-hw10
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9696 (Press CTRL+C to quit)
INFO:     172.17.0.1:43604 - "POST /predict HTTP/1.1" 200 OK
^CINFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [1]
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ docke rps
bash: docke: command not found
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ docker ps
CONTAINER ID   IMAGE                  COMMAND                  CREATED        STATUS          PORTS                       NAMES
a7cccc20110f   kindest/node:v1.27.3   "/usr/local/bin/entr…"   17 hours ago   Up 16 minutes   127.0.0.1:41847->6443/tcp   mlzoomcamp-control-plane
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ docker compose down
no configuration file provided: not found
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ docker stop mlzoomcamp-control-plane
mlzoomcamp-control-plane
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ kind create cluster
Creating cluster "kind" ...
 ✓ Ensuring node image (kindest/node:v1.27.3) 🖼
 ✓ Preparing nodes 📦  
 ✓ Writing configuration 📜 
 ✓ Starting control-plane 🕹️ 
 ✓ Installing CNI 🔌 
 ✓ Installing StorageClass 💾 
Set kubectl context to "kind-kind"
You can now use your cluster with:

kubectl cluster-info --context kind-kind

Have a nice day! 👋
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ kubectl cluster-info
Kubernetes control plane is running at https://127.0.0.1:45905
CoreDNS is running at https://127.0.0.1:45905/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ kubectl get services
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   12m
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ kind load docker-image zoomcamp-model:3.13.10-hw10
Image: "zoomcamp-model:3.13.10-hw10" with ID "sha256:8a3cee9bc8ba52289b5b01c6cca8f3bd23f8171a1f02cdc48e21fec252c4881a" not yet present on node "kind-control-plane", loading...
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ kubectl apply -f deployment.yaml
deployment.apps/subscription created
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ kubectl get pods
NAME                            READY   STATUS    RESTARTS   AGE
subscription-7469559794-8qcmf   1/1     Running   0          6s
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ kubectl apply -f service.yaml
service/subscription created
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ kubectl get svc
NAME           TYPE           CLUSTER-IP   EXTERNAL-IP   PORT(S)        AGE
kubernetes     ClusterIP      10.96.0.1    <none>        443/TCP        23m
subscription   LoadBalancer   10.96.49.3   <pending>     80:30233/TCP   8s
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ kubectl port-forward service/subscription 9696:80
Forwarding from 127.0.0.1:9696 -> 9696
Forwarding from [::1]:9696 -> 9696
Handling connection for 9696
```

### different terminal
```bash
@brukeg ➜ /workspaces/machine-learning-zoomcamp (main) $ cd ./machine-learning-zoomcamp/cohorts/2025/05-deployment/homework/
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ python q6_test.py
{'conversion_probability': 0.49999999999842815, 'conversion': False}
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ python q6_test.py
{'conversion_probability': 0.49999999999842815, 'conversion': False}
@brukeg ➜ .../cohorts/2025/05-deployment/homework (master) $ 
```


### deployment yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: subscription
spec:
  selector:
    matchLabels:
      app: subscription
  replicas: 1
  template:
    metadata:
      labels:
        app: subscription
    spec:
      containers:
      - name: subscription
        image: zoomcamp-model:3.13.10-hw10
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"            
          limits:
            memory: "512Mi"
            cpu: "500m"
        ports:
        - containerPort: 9696
```

### service yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: subscription
spec:
  type: LoadBalancer
  selector:
    app: subscription
  ports:
  - port: 80
    targetPort: 9696
```