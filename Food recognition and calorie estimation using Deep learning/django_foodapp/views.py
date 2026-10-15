from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.core.files.storage import FileSystemStorage
from .ml_model import predict_food

# HOME PAGE
def home(request):
    return render(request, 'home.html')
# CONTACT PAGE
def contact(request):
    return render(request, 'contact.html')
# ABOUT PAGE
def about(request):
    return render(request, 'about.html')

# SIGNUP
def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        User.objects.create_user(username=username, password=password)
        return redirect('login')

    return render(request, 'signup.html')

# LOGIN
def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect('upload')

    return render(request, 'login.html')

# LOGOUT
def logout_view(request):
    logout(request)
    return redirect('login')

# UPLOAD + PREDICT
def upload(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']

        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        file_path = fs.path(filename)

        food, prob, calories = predict_food(file_path)

        context = {
            'food': food,
            'prob': round(prob, 3),
            'calories': calories,
            'image_url': fs.url(filename)
        }
       

    return render(request, 'index.html', context)