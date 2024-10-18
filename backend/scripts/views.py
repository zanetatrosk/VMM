from .try_dogs import getPictureRecognition
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from PIL import Image

@api_view(['POST'])
def recognize_dog_breed(request):
    if 'file' not in request.FILES:
        response = Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)
        response["Access-Control-Allow-Origin"] = "*"
        return response

    file = request.FILES['file']
    try:
        img = Image.open(file)
    except IOError:
        response = Response({"error": "Invalid image file"}, status=status.HTTP_400_BAD_REQUEST)
        response["Access-Control-Allow-Origin"] = "*"
        return response

    results = getPictureRecognition(img)
    response = Response({"predictions": results}, status=status.HTTP_200_OK)
    response["Access-Control-Allow-Origin"] = "*"
    return response