apiVersion: apps/v1
kind: Deployment
metadata:
  name: transformer-train
spec:
  replicas: 1
  selector:
    matchLabels:
      app: transformer-train
  template:
    metadata:
      labels:
        app: transformer-train
    spec:
      containers:
        - name: transformer-train
          image: transformer-train:latest
          ports:
            - containerPort: 5000
          env:
            - name:
