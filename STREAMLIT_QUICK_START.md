# ⚡ QUICK START: Deploy to Streamlit in 5 Minutes

## ✅ Pre-Deployment Checklist (Do These Now)

### 1. Verify Everything is Pushed to GitHub
```bash
cd /workspaces/APS-Failure-Classifier
git status          # Should say "working tree clean"
git push origin main  # Push any remaining commits
```

### 2. Verify Model Files Exist in Repository
```bash
ls -lh outputs/aps_failure_model.joblib
ls -lh outputs/model_metrics.csv
ls -lh outputs/feature_importance.csv
```

**If files exist**: ✅ Continue to Streamlit setup

**If files missing**: ⚠️ You need to add them to GitHub
```bash
# Option A: Use Git LFS (for large files)
git lfs install
git lfs track "*.joblib"
git add outputs/
git commit -m "Add trained model artifact"
git push origin main

# Option B: Upload directly (if file < 100MB)
git add outputs/
git commit -m "Add trained model artifact"
git push origin main
```

### 3. Verify Key Files Exist
```bash
# These MUST exist in repo root:
✓ streamlit_app.py       (main entry point)
✓ app_logic.py           (ML pipeline)
✓ requirements.txt       (dependencies)
✓ .streamlit/config.toml (configuration)

# Check:
ls -la streamlit_app.py
ls -la requirements.txt
ls -la .streamlit/config.toml
```

---

## 🚀 Deploy to Streamlit Cloud (5 Steps)

### **STEP 1**: Go to Streamlit Cloud
```
https://share.streamlit.io/
```

### **STEP 2**: Log In with GitHub
- Click **"Sign up with GitHub"** OR **"Sign in"**
- Authorize Streamlit to access your GitHub account
- Grant **Read+Write** permissions

### **STEP 3**: Click "Create app" (Blue Button)
You'll see a form. Fill in:

```
Repository:        minahilahsanawan/APS-Failure-Classifier
Branch:            main
Main file path:    streamlit_app.py
```

**DO NOT CHANGE** any of these fields unless you named them differently!

### **STEP 4**: Click "Deploy"
- Streamlit starts building the app
- You'll see logs streaming
- **Wait 2-5 minutes** (don't close the page)

### **STEP 5**: App is Live! 🎉
You'll see a URL like:
```
https://aps-failure-classifier-xxxxx.streamlit.app/
```

**Copy this URL** to share, add to resume, etc.

---

## 🔍 Verify Deployment Worked

After Step 5, check:

1. **App loads**: Visit the URL in browser
2. **"Results" view shows**: 
   - Recall, Precision, PR-AUC, Cost reduction metrics
   - Cost comparison chart
   - Feature importance chart
3. **"Batch scoring" view works**:
   - Download template button works
   - File upload accepts CSV

If any of these fail → See Troubleshooting below ↓

---

## ❌ Troubleshooting

### Problem: "Error installing requirements"

**Solution A**: Update `requirements.txt`
```bash
cd /workspaces/APS-Failure-Classifier

# Option 1: Auto-generate from environment
pip freeze > requirements.txt

# Option 2: Manually verify these are present:
cat > requirements.txt << EOF
streamlit==1.59.1
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
joblib==1.4.2
altair==5.4.1
protobuf==4.25.1
EOF

git add requirements.txt
git commit -m "Fix: update requirements"
git push origin main
```

**Then**: Delete app on Streamlit (⋮ → Delete app) and redeploy

### Problem: "ModuleNotFoundError: app_logic"

**Solution**: Verify `streamlit_app.py` is in repo root
```bash
pwd  # Should be /workspaces/APS-Failure-Classifier
ls -la streamlit_app.py  # Should exist in current directory
```

### Problem: "FileNotFoundError: outputs/aps_failure_model.joblib"

**Solution**: Push model file to GitHub
```bash
# Check if file exists locally
ls -lh outputs/aps_failure_model.joblib

# If yes, push to GitHub
git add outputs/
git commit -m "Add model artifacts"
git push origin main

# If no, you need the trained model file (contact: minahilahsanawan)
```

### Problem: App Shows Error "No artifacts"

**Solution**: Check Streamlit logs
1. Go to: https://share.streamlit.io/
2. Click your app
3. Click **⋮ (three dots)** → **View logs**
4. Look for error messages starting with "ArtifactError:"
5. Common fix:
```bash
# Verify all files exist
ls outputs/aps_failure_model.joblib
ls outputs/model_metrics.csv
ls outputs/feature_importance.csv
ls .streamlit/config.toml

# Re-push if missing
git add .
git commit -m "Fix: add missing artifacts"
git push origin main
```

### Problem: Deployment Takes > 5 Minutes

**Reason**: Usually on first deploy (downloads + installs packages)

**Solution**:
- Wait up to 10 minutes
- Check **View logs** for progress
- If stuck > 10 min, try redeploying:
  1. Go to https://share.streamlit.io/
  2. Click **⋮** → **Reboot app**

---

## 📱 After Deployment: Using Your App

### On Mobile (Phone/Tablet)
1. Visit the URL: `https://aps-failure-classifier-xxxxx.streamlit.app/`
2. App is fully responsive ✅
3. Works on iOS Safari and Android Chrome

### Share with Others
- Send the full URL
- Works immediately (no login needed)
- Free tier supports ~50 concurrent users

### Update App
Make changes locally:
```bash
# Edit files
vim streamlit_app.py  # or open in VS Code

# Test locally
streamlit run streamlit_app.py

# Push to GitHub
git add .
git commit -m "Feature: describe your change"
git push origin main
```

**App auto-updates** on Streamlit Cloud (usually within 1-2 minutes)

---

## 📝 For Your Resume

```markdown
**APS Failure Classifier**
- Live Application: https://aps-failure-classifier.streamlit.app/
- Repository: https://github.com/minahilahsanawan/APS-Failure-Classifier
- Model Performance: 94.93% Recall, 0.994 ROC-AUC, 93% Cost Reduction
- Tech: Python, scikit-learn, Streamlit, pandas
- Deployed on: Streamlit Cloud
```

---

## ✅ Success Criteria

After following all steps, you should have:

- ✅ GitHub repo with all code committed and pushed
- ✅ Live Streamlit app at `https://aps-failure-classifier-xxxxx.streamlit.app/`
- ✅ App shows "Results" view with metrics
- ✅ App shows "Batch scoring" view with file upload
- ✅ No error messages or warnings
- ✅ URL works on desktop and mobile
- ✅ Can share URL with anyone

---

## 🆘 Still Having Issues?

1. **Check Streamlit logs**: https://share.streamlit.io/ → ⋮ → View logs
2. **Verify GitHub sync**: `git status` should show clean tree
3. **Test locally first**: `streamlit run streamlit_app.py`
4. **Check file paths**: All relative paths (no absolute paths like `C:\Users\...`)
5. **Verify requirements.txt**: Run `pip install -r requirements.txt` locally

---

**Need Help?**
- Streamlit Docs: https://docs.streamlit.io/streamlit-cloud/get-started
- Check if model files need Git LFS: https://git-lfs.github.com/

---

**🎉 READY TO DEPLOY? Follow the 5 steps above!**
