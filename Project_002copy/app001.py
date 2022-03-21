from flask_forms001 import app,db
from flask import render_template,redirect,request,url_for,flash
from flask_login import login_user,login_required,logout_user
from flask_forms001.models import User
from forms import LoginForm, RegistrationForm


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/welcome')
@login_required  #makes sure the user is logged in to see the view below
def welcome_user():
    return render_template('welcome_user.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You are logged out!")
    return redirect(url_for('home'))

@app.route('/login',methods=['GET','POST'])
def login():        #setting up login view
     form=LoginForm()   #creating login form
     if form.validate_on_submit():  #check if valid on submission
         user=User.query.filter_by(email=form.email.data).first()  #grabbing user based on email

         if user.check_password(form.password.data) and user is not None: #pw correct and user exists
             login_user(user)
             flash('Logged in Successfully!')

             next=request.args.get('next') #when user tries to access a page that requires login, flask saves it as next page and redirects to login and after login, directed to the req. page
             if next== None or not next[0]=='/': #check if next exists else go to welcome page
                 next=url_for('welcome_user')

             return redirect(next)

     return render_template('login001.html',form=form)


@app.route('/register',methods=['GET','POST'])
def register():
    form=RegistrationForm()

    if form.validate_on_submit():
        user=User(email=form.email.data,username=form.username.data,
                  password=form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Thanks for registration")
        return redirect(url_for('login'))
    return render_template('register001.html',form=form)

if __name__=='__main__':
    app.run(debug=True)
