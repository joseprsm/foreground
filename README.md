# Foreground

Foreground is a fork from [xuebinqin/DIS](https://github.com/xuebinqin/DIS).

Essentially it's a containerised service around the DIS model, using the 
model weights available at the Google Drive location specified by the authors.

You can build the docker image yourself, or use the one publicly
available at `joseprsm/foreground`. 

```shell
export IMAGE_URI=joseprsm/foreground  # replace by your image URI
docker build . -t $IMAGE_URI  # not required if using the default image URI
```

To run the service, you just have to run:

```shell
docker run -p 8000:8000 $IMAGE_URI
```

This will bind port 8000 to the container's. You can then send POST requests 
to the model. Notice the key to input file must be `file`.

```shell
curl -i -X POST -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" http://localhost:8000
```