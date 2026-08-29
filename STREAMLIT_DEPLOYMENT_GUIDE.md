# 🚀 Streamlit Cloud Deployment Guide

**Complete step-by-step guide to deploy the APS Failure Classifier on Streamlit Cloud**

---

## Step 1: Prepare Your GitHub Repository ✅

### 1.1 Verify Repository is Public
1. Go to: https://github.com/minahilahsanawan/APS-Failure-Classifier
2. Click **Settings** (gear icon)
3. Scroll down to **Danger Zone** → **Repository visibility**
4. Ensure it's set to **Public** (required for Streamlit Cloud free tier)
5. If Private: Click **Change visibility** → **Make public** → Confirm

### 1.2 Verify All Files Are Committed
```bash
cd /workspaces/APS-Failure-Classifier
git status  # Should show "nothing to commit, working tree clean"
git log --oneline -5  # View recent commits
```

### 1.3 Verify Repository Structure
Your GitHub repo should contain:
```
APS-Failure-Classifier/
├── streamlit_app.py              ← Main app file (MUST be named exactly this)
├── app_logic.py                  ← Core ML logic
├── test_app_logic.py             ← Tests
├── requirements.txt              ← Python dependencies
├── aps_failure_eda_model.ipynb   ← Model notebook
├── .streamlit/
│   └── config.toml               ← Streamlit config
├── outputs/
│   ├── aps_failure_model.joblib  ← Trained model
│   ├── model_metrics.csv         ← Metrics
│   └── feature_importance.csv    ← Feature importance
└── README.md                     ← Documentation
```

**⚠️ IMPORTANT**: All trained model files (`.joblib`, `.csv`) MUST be in the repo!

---

## Step 2: Create Streamlit Cloud Account ✅

### 2.1 Sign Up for Streamlit Cloud
1. Go to: https://streamlit.io/cloud
2. Click **Sign up**
3. Choose: **Sign up with GitHub** (recommended)
4. Authorize Streamlit to access your GitHub account
5. Grant permissions when prompted

### 2.2 Configure GitHub App Access
1. After signing in, go to: https://share.streamlit.io/
2. Click your profile (top right) → **Settings**
3. Under **Installed apps** → Verify "Streamlit" is authorized
4. If not: Click **Link GitHub** and authorize again

---

## Step 3: Deploy Your App on Streamlit Cloud 🚀

### 3.1 Create New App
1. Go to: https://share.streamlit.io/
2. Click **Create app** (blue button, top right)
3. You'll see a form:

```
Repository:        minahilahsanawan/APS-Failure-Classifier
Branch:            main
Main file path:    streamlit_app.py
```

4. **IMPORTANT**: Set Main file path to: `streamlit_app.py` (exact filename)

### 3.2 Click "Deploy" Button
- Streamlit will:
  1. Clone your GitHub repo
  2. Install packages from `requirements.txt`
  3. Load the model from `outputs/aps_failure_model.joblib`
  4. Start the Streamlit server
  5. Generate a public URL

### 3.3 Wait for Deployment (2-5 minutes)
You'll see deployment logs:
```
Building image...
Installing dependencies...
Pushing image to registry...
Starting container...
Running streamlit_app.py...
```

✅ When complete: **App is LIVE!**

---

## Step 4: Access Your Deployed App

After successful deployment, you'll get a URL like:
```
https://aps-failure-classifier-xxxxx.streamlit.app/
```

### 4.1 Share Your App
- Full URL: `https://aps-failure-classifier-xxxxx.streamlit.app/`
- For resumes/MITACS: Share this URL
- Works on desktop AND mobile
- Anyone with the link can use it

---

## Step 5: Fix Errors (If App Doesn't Load) 🔧

### 5.1 Check Deployment Logs
1. Go to: https://share.streamlit.io/
2. Find your app in the list
3. Click **⋮ (three dots)** → **View logs**
4. Look for error messages

### 5.2 Common Errors & Fixes

#### Error: "FileNotFoundError: outputs/aps_failure_model.joblib"
**Cause**: Model file not in GitHub repo
**Fix**:
```bash
# Ensure model file is in repo
ls -lh outputs/aps_failure_model.joblib

# If not, you need to add it (but file might be too large for Git)
# Use Git LFS instead:
git lfs install
git lfs track "*.joblib"
git add outputs/aps_failure_model.joblib
git commit -m "Add trained model"
git push
```

#### Error: "ModuleNotFoundError: No module named 'pandas'"
**Cause**: Missing dependency in `requirements.txt`
**Fix**:
```bash
# Update requirements.txt
pip freeze > requirements.txt

# Or manually add:
# streamlit==1.59.1
# pandas==2.2.3
# numpy==1.26.4
# scikit-learn==1.5.2
# joblib==1.4.2
# altair==5.4.1
# protobuf==4.25.1

git add requirements.txt
git commit -m "Fix: add missing dependencies"
git push
```

#### Error: "Timeout while loading app"
**Cause**: App takes too long to start (> 30 seconds)
**Fix**:
```bash
# Streamlit config: .streamlit/config.toml
[client]
showWarningOnDirectExecution = false

[server]
maxUploadSize = 200
```

#### Error: "Permission denied on deployment"
**Cause**: GitHub app not properly authorized
**Fix**:
1. Log out of https://share.streamlit.io/
2. Go to: https://github.com/settings/applications
3. Find "Streamlit" → Click **Revoke**
4. Log back into https://share.streamlit.io/
5. Re-authorize GitHub access

---

## Step 6: Keep App Updated 📝

### 6.1 Update Code
```bash
# Make changes locally
# Edit streamlit_app.py or app_logic.py

# Test locally
streamlit run streamlit_app.py

# Commit and push
git add .
git commit -m "Feature: Add new functionality"
git push origin main
```

### 6.2 App Auto-Updates
- Streamlit Cloud **automatically redeploys** when you push to GitHub
- Check deployment status: https://share.streamlit.io/ → Click your app
- Usually takes 1-2 minutes

---

## Step 7: Configure App Settings (Optional)

### 7.1 Customize Streamlit Config
File: `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#6366f1"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#161b22"
textColor = "#c9d1d9"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true

[logger]
level = "info"
```

### 7.2 Add Custom Favicon
```toml
[browser]
gatherUsageStats = false
```

---

## Step 8: Monitor Your App 📊

### 8.1 View App Analytics
1. Go to: https://share.streamlit.io/
2. Find your app → Click app name
3. See: **Views**, **Users**, **Duration**

### 8.2 Check App Health
- Look for red/yellow badges indicating issues
- Click app to view detailed logs
- Monitor error rate

### 8.3 Set Up Email Alerts (Optional)
- Streamlit sends email if app crashes
- Check email for deployment notifications

---

## Troubleshooting Checklist ✓

- [ ] Repository is **Public** on GitHub
- [ ] All files committed and pushed
- [ ] `streamlit_app.py` exists in root directory
- [ ] `requirements.txt` lists all dependencies
- [ ] Model file `outputs/aps_failure_model.joblib` is in repo
- [ ] `.streamlit/config.toml` exists and is valid
- [ ] Streamlit GitHub app is authorized in your account
- [ ] Branch is set to `main` (or correct branch)
- [ ] No special characters in file paths
- [ ] All imports in Python files are valid

---

## Quick Reference Commands

```bash
# Check app status locally
streamlit run streamlit_app.py

# Test requirements.txt
pip install -r requirements.txt

# View git history
git log --oneline -5

# Verify all files
ls -la outputs/
ls -la .streamlit/

# Check repo status
git status
git remote -v  # Should show GitHub URL
```

---

## Support Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Streamlit Forum**: https://discuss.streamlit.io/
- **GitHub Issues**: https://github.com/streamlit/streamlit/issues
- **Deployment Docs**: https://docs.streamlit.io/streamlit-cloud/get-started

---

## Final Deployment URL

After deployment, share this format:
```
🚀 Live App: https://aps-failure-classifier.streamlit.app/
```

**On your resume**, add:
```
Live Application: https://aps-failure-classifier.streamlit.app/
Repository: https://github.com/minahilahsanawan/APS-Failure-Classifier
```

---

**Status**: Ready for Production ✅

All steps complete = Professional deployment on Streamlit Cloud!
