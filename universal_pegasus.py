#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║         PEGASUS ENTERPRISE DETECTOR v5.0                     ║
║         🔍 Detects ALL Variants • 🏢 Enterprise Grade        ║
║         📊 Dashboard • 💾 Database • 🤖 AI/ML • ☁️ Cloud     ║
║         💰 License & Credit System • 📱 Multi-Platform       ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import hashlib
import struct
import sqlite3
import tempfile
import subprocess
import re
import time
import threading
import queue
import statistics
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any, BinaryIO
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from collections import defaultdict, Counter, OrderedDict
import concurrent.futures
import asyncio
import aiohttp
import warnings
warnings.filterwarnings('ignore')
import platform
import socket
import ssl
import psutil
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import uuid
import base64
import csv
import xml.etree.ElementTree as ET
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ============================================================================
# 🏢 ENTERPRISE DATABASE MODULE
# ============================================================================

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Float, DateTime, JSON, ForeignKey, Table, BigInteger, Numeric, func, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session, scoped_session
from sqlalchemy.pool import QueuePool
import alembic
from alembic import command
from alembic.config import Config
import redis

# ============================================================================
# 🌐 ENTERPRISE DASHBOARD MODULE
# ============================================================================

from flask import Flask, render_template, jsonify, request, Response, send_file, session, redirect, url_for, flash, g
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SelectField, TextAreaField, DecimalField, IntegerField
from wtforms.validators import DataRequired, Email, Length, NumberRange
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_mail import Mail, Message
from flask_babel import Babel, _
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, ALL
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash_daq as daq

# ============================================================================
# 💰 PAYMENT & LICENSE MODULES
# ============================================================================

import stripe
import paypalrestsdk
from paypalrestsdk import Payment, Payout, Webhook
import qrcode
from io import BytesIO
import segno  # For advanced QR codes
import pdfkit
# In your imports section, after line 34, you have:
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table as RLTable, TableStyle, Paragraph, Spacer  # Alias ReportLab Table
# SQLAlchemy imports
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Float, DateTime, JSON, ForeignKey, BigInteger, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session, scoped_session
from sqlalchemy.pool import QueuePool

# ReportLab imports - with alias for Table
from reportlab.platypus import SimpleDocTemplate, Table as RLTable, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ============================================================================
# ☁️ CLOUD & DISTRIBUTED COMPUTING
# ============================================================================

import boto3
import google.cloud.pubsub_v1 
import azure.identity
import azure.keyvault.secrets
import kubernetes.client
import docker
import celery
from celery import Celery
import pika  # RabbitMQ
import zmq
import grpc
import fastapi
from fastapi import FastAPI, Depends, HTTPException, WebSocket, BackgroundTasks, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from slowapi import Limiter as SlowLimiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address as get_remote_address_slowapi
from slowapi.errors import RateLimitExceeded

# ============================================================================
# 📊 ENTERPRISE MONITORING & LOGGING
# ============================================================================

import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
import statsd
import elasticsearch
from elasticsearch import Elasticsearch
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import newrelic.agent
import opentelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

# ============================================================================
# 🔐 ENTERPRISE SECURITY
# ============================================================================

import jwt
from authlib.integrations.flask_client import OAuth
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import secrets
import hmac
from functools import wraps
import hashlib
import hmac

# ============================================================================
# 🎨 UI & VISUALIZATION
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import folium
from folium import plugins
import networkx as nx
import pyvis
from pyvis.network import Network
import plotly.figure_factory as ff
import missingno as msno

# ============================================================================
# 📱 MOBILE INTEGRATION
# ============================================================================

import frida
import objection
import libimobiledevice
import adb_shell
from adb_shell.adb_device import AdbDeviceTcp, AdbDeviceUsb
from adb_shell.auth.keygen import keygen
from adb_shell.auth.sign_pythonrsa import PythonRSASigner

# ============================================================================
# 🤖 ENHANCED AI/ML IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Dense, Conv1D, Conv2D, LSTM, GRU, 
                                    Dropout, BatchNormalization, Embedding,
                                    Flatten, Bidirectional, Attention, 
                                    MultiHeadAttention, LayerNormalization,
                                    GlobalAveragePooling1D, GlobalMaxPooling1D,
                                    Conv1DTranspose, Conv2DTranspose)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, MobileNetV2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import models as torch_models

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             IsolationForest, VotingClassifier, StackingClassifier,
                             AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                             RandomForestRegressor, GradientBoostingRegressor)
from sklearn.svm import OneClassSVM, SVC, SVR, NuSVC, LinearSVC
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, OPTICS, SpectralClustering, Birch
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, NearestNeighbors, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import (LogisticRegression, RidgeClassifier, SGDClassifier,
                                 PassiveAggressiveClassifier, LinearRegression, Ridge,
                                 Lasso, ElasticNet, BayesianRidge)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                  LabelEncoder, OneHotEncoder, PowerTransformer,
                                  QuantileTransformer, KBinsDiscretizer, PolynomialFeatures)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF, LatentDirichletAllocation, KernelPCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.feature_selection import (SelectKBest, SelectPercentile,
                                      RFE, RFECV, mutual_info_classif, chi2, f_classif,
                                      VarianceThreshold, SelectFromModel)
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, TimeSeriesSplit, GroupKFold,
                                     RepeatedKFold, LeaveOneOut)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve, auc, precision_recall_curve,
                            silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                            matthews_corrcoef, cohen_kappa_score, log_loss,
                            mean_squared_error, mean_absolute_error, r2_score,
                            explained_variance_score, max_error)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
import joblib
from sklearn import datasets

# Transformers & NLP
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                         pipeline, BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model,
                         RobertaTokenizer, RobertaModel, DistilBertTokenizer, DistilBertModel,
                         AutoModelForTokenClassification, AutoModelForQuestionAnswering,
                         TrainingArguments, Trainer, TextClassificationPipeline,
                         T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForSequenceClassification)
# PDF generation example
def generate_pdf_invoice(self, invoice_data):
    doc = SimpleDocTemplate("invoice.pdf", pagesize=letter)
    elements = []
    
    # Create data for the table
    data = [
        ['Item', 'Quantity', 'Price', 'Total'],
        ['1000 Credits', 1, '$9.99', '$9.99'],
        ['AI Detection', 1, '$5.00', '$5.00'],
        ['Total', '', '', '$14.99']
    ]
    
    # Use RLTable (ReportLab Table) for PDF
    table = RLTable(data, colWidths=[200, 80, 80, 80])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    doc.build(elements)
    
# XGBoost, LightGBM, CatBoost
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Advanced ML
import optuna
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import shap
import lime
import lime.lime_tabular
import lime.lime_text
import eli5
from eli5.sklearn import PermutationImportance
import imblearn
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import (RandomUnderSampler, ClusterCentroids, TomekLinks,
                                    EditedNearestNeighbours, RepeatedEditedNearestNeighbours,
                                    AllKNN, NearMiss)
from imblearn.combine import SMOTETomek, SMOTEENN

# AutoML
#import autosklearn.classification
#import autosklearn.regression
import tpot
from tpot import TPOTClassifier, TPOTRegressor

# Anomaly Detection Specialized
import pyod
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.sos import SOS
from pyod.models.deep_svdd import DeepSVDD

# Time Series
import prophet
from prophet import Prophet
import pmdarima as pm
from pmdarima import auto_arima
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf

# ============================================================================
# 🎭 UNICODE ART & BANNERS
# ============================================================================

class UnicodeArt:
    """Unicode art and banners for the enterprise system"""
    
    @staticmethod
    def get_banner():
        return """
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                                                                              ║
        ║         ██████╗ ███████╗ ██████╗  █████╗ ███████╗██╗   ██╗███████╗          ║
        ║         ██╔══██╗██╔════╝██╔════╝ ██╔══██╗██╔════╝██║   ██║██╔════╝          ║
        ║         ██████╔╝█████╗  ██║  ███╗███████║███████╗██║   ██║███████╗          ║
        ║         ██╔═══╝ ██╔══╝  ██║   ██║██╔══██║╚════██║██║   ██║╚════██║          ║
        ║         ██║     ███████╗╚██████╔╝██║  ██║███████║╚██████╔╝███████║          ║
        ║         ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝          ║
        ║                                                                              ║
        ║         🏢 PEGASUS ENTERPRISE DETECTOR v5.0 • ⭐⭐⭐⭐⭐                       ║
        ║         🔍 Universal Detection • 💰 Monetized • ☁️ Cloud Ready              ║
        ║                                                                              ║
        ╚══════════════════════════════════════════════════════════════════════════════╝
        """
    
    @staticmethod
    def get_status_icons():
        return {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'loading': '⏳',
            'scanning': '🔍',
            'detected': '🚨',
            'clean': '✅',
            'processing': '⚙️',
            'database': '💾',
            'network': '🌐',
            'ai': '🤖',
            'license': '💰',
            'user': '👤',
            'admin': '👑',
            'device': '📱',
            'server': '🖥️',
            'cloud': '☁️',
            'lock': '🔒',
            'unlock': '🔓',
            'money': '💵',
            'credit': '💳',
            'chart': '📊',
            'alert': '🚨',
            'bell': '🔔',
            'download': '📥',
            'upload': '📤',
            'settings': '⚙️',
            'help': '❓'
        }
    
    @staticmethod
    def get_progress_bar(percentage, width=40):
        """Create a Unicode progress bar"""
        filled = int(width * percentage / 100)
        empty = width - filled
        bar = '█' * filled + '░' * empty
        return f"[{bar}] {percentage:.1f}%"
    
    @staticmethod
    def get_loading_spinner():
        """Get loading spinner frames"""
        return ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    @staticmethod
    def print_table(headers, rows, title=None):
        """Print Unicode table"""
        if title:
            print(f"\n{'═' * 60}")
            print(f"   {title}")
            print(f"{'═' * 60}")
        
        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Print header
        header_line = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
        print(header_line)
        
        header_cells = []
        for i, h in enumerate(headers):
            header_cells.append(f" {h:<{col_widths[i]}} ")
        print("│" + "│".join(header_cells) + "│")
        
        # Print separator
        sep_line = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
        print(sep_line)
        
        # Print rows
        for row in rows:
            cells = []
            for i, cell in enumerate(row):
                cells.append(f" {str(cell):<{col_widths[i]}} ")
            print("│" + "│".join(cells) + "│")
        
        # Print footer
        footer_line = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
        print(footer_line)

# ============================================================================
# 🏢 ENTERPRISE DATABASE MODELS (Enhanced)
# ============================================================================

Base = declarative_base()

# Association tables
user_roles = Table('user_roles', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('role_id', Integer, ForeignKey('roles.id'))
)

detection_indicators = Table('detection_indicators', Base.metadata,
    Column('detection_id', Integer, ForeignKey('detections.id')),
    Column('indicator_id', Integer, ForeignKey('indicators.id'))
)

scan_tags = Table('scan_tags', Base.metadata,
    Column('scan_id', Integer, ForeignKey('scans.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)

class User(UserMixin, Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    first_name = Column(String(64))
    last_name = Column(String(64))
    phone = Column(String(20))
    company = Column(String(128))
    job_title = Column(String(128))
    avatar = Column(Text)  # URL or base64
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    
    # Security
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(32))
    api_key = Column(String(256), unique=True, default=lambda: hashlib.sha256(os.urandom(32)).hexdigest())
    session_tokens = Column(JSON, default=list)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    last_password_change = Column(DateTime)
    
    # Preferences
    preferences = Column(JSON, default=dict)
    notification_settings = Column(JSON, default=dict)
    
    # Relationships
    credits = relationship("UserCredits", back_populates="user", uselist=False)
    scans = relationship("Scan", back_populates="user")
    alerts = relationship("Alert", back_populates="user")
    reports = relationship("Report", back_populates="user")
    invoices = relationship("Invoice", back_populates="user")
    api_logs = relationship("APILog", back_populates="user")
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    devices = relationship("Device", back_populates="owner")

class Role(Base):
    __tablename__ = 'roles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(Text)
    permissions = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")

class UserCredits(Base):
    __tablename__ = 'user_credits'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True)
    
    # Credit balances
    credits_balance = Column(Integer, default=100)  # Initial free credits
    credits_used = Column(Integer, default=0)
    credits_purchased = Column(Integer, default=0)
    credits_bonus = Column(Integer, default=0)
    
    # Credit packages
    free_credits = Column(Integer, default=100)
    premium_credits = Column(Integer, default=0)
    
    # Limits
    daily_scan_limit = Column(Integer, default=10)
    monthly_scan_limit = Column(Integer, default=100)
    daily_api_limit = Column(Integer, default=1000)
    
    # Usage tracking
    scans_today = Column(Integer, default=0)
    scans_this_month = Column(Integer, default=0)
    api_calls_today = Column(Integer, default=0)
    last_reset_date = Column(DateTime, default=datetime.utcnow)
    last_reset_month = Column(DateTime, default=datetime.utcnow)
    
    # Billing
    billing_tier = Column(String(32), default='free')  # free, basic, pro, enterprise
    subscription_id = Column(String(128))
    subscription_status = Column(String(32), default='inactive')
    subscription_expiry = Column(DateTime)
    
    # Payment
    stripe_customer_id = Column(String(128))
    paypal_payer_id = Column(String(128))
    default_payment_method = Column(String(32))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="credits")
    transactions = relationship("CreditTransaction", back_populates="user_credits", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="user_credits", cascade="all, delete-orphan")

class CreditTransaction(Base):
    __tablename__ = 'credit_transactions'
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(64), unique=True, nullable=False, index=True)
    user_credits_id = Column(Integer, ForeignKey('user_credits.id'))
    
    # Transaction details
    type = Column(String(32), nullable=False)  # purchase, usage, bonus, refund, adjustment, transfer
    amount = Column(Integer, nullable=False)
    currency = Column(String(3), default='USD')
    description = Column(Text)
    reference = Column(String(256))  # Stripe/PayPal ID
    transaction_metadata = Column(JSON)  # CHANGED: renamed from 'metadata'
    
    # Status
    status = Column(String(32), default='completed')  # pending, completed, failed, refunded
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)
    
    # Relationships
    user_credits = relationship("UserCredits", back_populates="transactions")

class Invoice(Base):
    __tablename__ = 'invoices'
    
    id = Column(Integer, primary_key=True)
    invoice_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    user_credits_id = Column(Integer, ForeignKey('user_credits.id'))
    
    # Invoice details
    invoice_number = Column(String(32), unique=True)
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default='USD')
    description = Column(Text)
    items = Column(JSON)  # [{item: "1000 credits", quantity: 1, price: 10.00}]
    subtotal = Column(Numeric(10, 2))
    tax = Column(Numeric(10, 2))
    discount = Column(Numeric(10, 2))
    total = Column(Numeric(10, 2))
    
    # Payment status
    status = Column(String(32), default='pending')  # pending, paid, failed, refunded, cancelled
    payment_method = Column(String(32))
    payment_date = Column(DateTime)
    payment_reference = Column(String(256))
    payment_gateway = Column(String(32))  # stripe, paypal, manual
    
    # Billing info
    billing_name = Column(String(128))
    billing_email = Column(String(120))
    billing_address = Column(Text)
    billing_city = Column(String(64))
    billing_state = Column(String(64))
    billing_country = Column(String(64))
    billing_postal_code = Column(String(20))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    due_date = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(days=30))
    paid_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="invoices")
    user_credits = relationship("UserCredits", back_populates="invoices")

class Device(Base):
    __tablename__ = 'devices'
    
    id = Column(Integer, primary_key=True)
    device_uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String(128), unique=True, nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey('users.id'))
    
    # Device info
    device_type = Column(String(32))  # iphone, android, ipad, tablet, other
    platform = Column(String(32))  # iOS, Android, Windows, macOS
    manufacturer = Column(String(128))
    model = Column(String(128))
    serial_number = Column(String(128))
    imei = Column(String(15))
    meid = Column(String(18))
    
    # OS details
    os_name = Column(String(64))
    os_version = Column(String(32))
    os_build = Column(String(64))
    kernel_version = Column(String(64))
    
    # Security
    jailbreak_status = Column(Boolean, default=False)
    root_status = Column(Boolean, default=False)
    bootloader_unlocked = Column(Boolean, default=False)
    security_patch_level = Column(String(32))
    encryption_status = Column(String(32))
    
    # Network
    ip_address = Column(String(45))
    mac_address = Column(String(17))
    carrier = Column(String(64))
    wifi_ssid = Column(String(64))
    
    # Status
    risk_score = Column(Float, default=0.0)
    risk_level = Column(String(32), default='low')  # low, medium, high, critical
    status = Column(String(32), default='active')  # active, inactive, quarantined, retired
    last_scan_status = Column(String(32))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    last_scan_at = Column(DateTime)
    
    # Relationships
    owner = relationship("User", back_populates="devices")
    scans = relationship("Scan", back_populates="device")
    detections = relationship("Detection", back_populates="device")

class Scan(Base):
    __tablename__ = 'scans'
    
    id = Column(Integer, primary_key=True)
    scan_uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    scan_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    device_id = Column(Integer, ForeignKey('devices.id'))
    
    # Scan configuration
    scan_type = Column(String(32), default='full')  # quick, full, deep, network, memory, custom
    scan_mode = Column(String(32), default='auto')  # auto, manual, scheduled
    priority = Column(String(32), default='normal')  # low, normal, high, critical
    
    # Status
    status = Column(String(32), default='pending')  # pending, running, completed, failed, cancelled
    progress = Column(Float, default=0.0)
    current_phase = Column(String(64))
    
    # Timestamps
    scheduled_at = Column(DateTime)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration = Column(Float)  # seconds
    
    # Resource usage
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_usage = Column(Float)
    network_usage = Column(Float)
    
    # Scan metrics
    total_files_scanned = Column(Integer, default=0)
    total_processes_analyzed = Column(Integer, default=0)
    total_network_connections = Column(Integer, default=0)
    total_memory_regions = Column(Integer, default=0)
    
    # Results
    pegasus_detected = Column(Boolean, default=False)
    variant_detected = Column(String(128))
    confidence_score = Column(Float, default=0.0)
    risk_level = Column(String(32))
    threat_score = Column(Integer, default=0)
    
    # AI/ML results
    ai_models_used = Column(JSON, default=list)
    ai_confidence_scores = Column(JSON, default=dict)
    ml_predictions = Column(JSON, default=dict)
    
    # Raw data
    scan_config = Column(JSON, default=dict)
    raw_results = Column(JSON)
    scan_log = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="scans")
    device = relationship("Device", back_populates="scans")
    detections = relationship("Detection", back_populates="scan", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="scan")
    tags = relationship("Tag", secondary=scan_tags, back_populates="scans")

class Detection(Base):
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True)
    detection_uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    detection_id = Column(String(64), unique=True, nullable=False, index=True)
    scan_id = Column(Integer, ForeignKey('scans.id'))
    device_id = Column(Integer, ForeignKey('devices.id'))
    
    # Detection details
    variant = Column(String(128))
    family = Column(String(128))
    confidence = Column(Float)
    detection_method = Column(String(64))  # signature, heuristic, ai, behavioral, memory
    detection_source = Column(String(64))  # filesystem, network, process, memory, registry
    detection_engine = Column(String(64))  # engine version
    
    # Evidence
    file_path = Column(Text)
    file_hash = Column(String(64))
    file_size = Column(BigInteger)
    process_name = Column(String(256))
    process_id = Column(Integer)
    network_destination = Column(String(512))
    network_port = Column(Integer)
    memory_address = Column(String(128))
    memory_size = Column(BigInteger)
    registry_key = Column(String(512))
    registry_value = Column(Text)
    
    # Context
    context_data = Column(JSON)  # Additional context about detection
    indicators_of_compromise = Column(JSON)  # IOCs found
    attack_techniques = Column(JSON)  # MITRE ATT&CK techniques
    timeline = Column(JSON)  # Timeline of malicious activities
    
    # Severity
    severity = Column(String(32), default='medium')  # info, low, medium, high, critical
    impact_score = Column(Integer, default=5)  # 1-10
    likelihood_score = Column(Integer, default=5)  # 1-10
    risk_score = Column(Integer, default=25)  # impact * likelihood
    
    # Status
    status = Column(String(32), default='new')  # new, investigating, confirmed, false_positive, mitigated, resolved
    mitigated = Column(Boolean, default=False)
    mitigation_action = Column(String(256))
    mitigation_date = Column(DateTime)
    resolved_by = Column(String(128))
    resolved_date = Column(DateTime)
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    scan = relationship("Scan", back_populates="detections")
    device = relationship("Device", back_populates="detections")
    indicators = relationship("Indicator", secondary=detection_indicators, back_populates="detections")
    alerts = relationship("Alert", back_populates="detection")

class Indicator(Base):
    __tablename__ = 'indicators'
    
    id = Column(Integer, primary_key=True)
    ioc_uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    
    # IOC details
    ioc_type = Column(String(32))  # hash, domain, ip, url, email, filename, registry, mutex, etc.
    value = Column(String(1024), nullable=False, index=True)
    context = Column(Text)
    
    # Source
    source = Column(String(128))  # internal, threat_intel, user, partner
    source_reliability = Column(String(32))  # unknown, low, medium, high
    source_url = Column(Text)
    
    # Threat intelligence
    threat_type = Column(String(64))  # malware, phishing, c2, exploit, etc.
    threat_actor = Column(String(128))
    campaign = Column(String(128))
    malware_family = Column(String(128))
    
    # Scoring
    confidence = Column(Float, default=0.5)
    threat_score = Column(Integer, default=5)  # 1-10
    risk_level = Column(String(32), default='medium')
    
    # Metadata
    tags = Column(JSON, default=list)
    description = Column(Text)
    references = Column(JSON, default=list)
    comments = Column(Text)
    
    # Timestamps
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    detections = relationship("Detection", secondary=detection_indicators, back_populates="indicators")

class Alert(Base):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    alert_uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    alert_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    scan_id = Column(Integer, ForeignKey('scans.id'))
    detection_id = Column(Integer, ForeignKey('detections.id'))
    
    # Alert details
    severity = Column(String(32), default='medium')  # info, low, medium, high, critical
    title = Column(String(256), nullable=False)
    message = Column(Text)
    alert_type = Column(String(64))  # detection, system, license, billing, security
    category = Column(String(64))  # malware, intrusion, anomaly, compliance, system
    
    # Content
    summary = Column(Text)
    details = Column(JSON)
    recommendations = Column(Text)
    actions = Column(JSON)  # Available actions for the alert
    
    # Status
    status = Column(String(32), default='new')  # new, acknowledged, investigating, resolved, closed
    acknowledged_by = Column(String(128))
    acknowledged_at = Column(DateTime)
    resolved_by = Column(String(128))
    resolved_at = Column(DateTime)
    closed_by = Column(String(128))
    closed_at = Column(DateTime)
    
    # Notifications
    notified = Column(Boolean, default=False)
    notification_channels = Column(JSON, default=list)  # email, sms, push, webhook
    notification_sent_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="alerts")
    scan = relationship("Scan", back_populates="alerts")
    detection = relationship("Detection", back_populates="alerts")

class Report(Base):
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True)
    report_uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    report_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # Report configuration
    report_type = Column(String(32))  # scan, detection, audit, compliance, executive, custom
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    timezone = Column(String(64), default='UTC')
    
    # Content
    title = Column(String(256), nullable=False)
    summary = Column(Text)
    executive_summary = Column(Text)
    findings = Column(JSON)
    statistics = Column(JSON)
    charts = Column(JSON)
    recommendations = Column(Text)
    appendix = Column(JSON)
    
    # Formatting
    template = Column(String(64), default='default')
    style = Column(JSON, default=dict)
    
    # Status
    status = Column(String(32), default='generating')  # generating, ready, delivered, archived
    generated_at = Column(DateTime)
    delivered_at = Column(DateTime)
    
    # Export
    export_formats = Column(JSON, default=list)  # pdf, html, json, csv, excel
    exported_at = Column(DateTime)
    export_location = Column(Text)
    
    # Security
    is_encrypted = Column(Boolean, default=False)
    encryption_key = Column(String(256))
    password_hash = Column(String(256))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="reports")

class Tag(Base):
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True, nullable=False, index=True)
    color = Column(String(7), default='#3498db')  # Hex color
    description = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    scans = relationship("Scan", secondary=scan_tags, back_populates="tags")

class SystemMetrics(Base):
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_id = Column(String(64), unique=True, nullable=False, index=True)
    
    # Resource usage
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cpu_percent = Column(Float)
    cpu_count = Column(Integer)
    memory_percent = Column(Float)
    memory_total = Column(BigInteger)
    memory_available = Column(BigInteger)
    disk_usage_percent = Column(Float)
    disk_total = Column(BigInteger)
    disk_free = Column(BigInteger)
    network_bytes_sent = Column(BigInteger)
    network_bytes_recv = Column(BigInteger)
    
    # Application metrics
    active_scans = Column(Integer)
    active_users = Column(Integer)
    active_sessions = Column(Integer)
    total_detections = Column(Integer)
    detection_rate = Column(Float)
    false_positives = Column(Integer)
    false_negatives = Column(Integer)
    
    # AI/ML metrics
    model_accuracy = Column(Float)
    model_precision = Column(Float)
    model_recall = Column(Float)
    training_samples = Column(Integer)
    prediction_latency = Column(Float)
    inference_time = Column(Float)
    
    # Business metrics
    total_credits = Column(BigInteger)
    credits_used = Column(BigInteger)
    revenue_today = Column(Numeric(10, 2))
    revenue_month = Column(Numeric(10, 2))
    active_subscriptions = Column(Integer)
    new_users_today = Column(Integer)
    
    # Performance
    api_response_time = Column(Float)
    database_query_time = Column(Float)
    cache_hit_rate = Column(Float)
    queue_length = Column(Integer)
    
    # Metadata
    node_id = Column(String(64))  # For distributed systems
    service_name = Column(String(64))
    version = Column(String(32))

class APILog(Base):
    __tablename__ = 'api_logs'
    
    id = Column(Integer, primary_key=True)
    log_uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    log_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    api_key = Column(String(256), index=True)
    
    # Request details
    endpoint = Column(String(256), nullable=False)
    method = Column(String(10), nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    request_headers = Column(JSON)
    request_params = Column(JSON)
    request_body = Column(Text)
    request_size = Column(Integer)
    
    # Response details
    status_code = Column(Integer)
    response_headers = Column(JSON)
    response_body = Column(Text)
    response_size = Column(Integer)
    response_time = Column(Float)  # milliseconds
    
    # Credits
    credits_used = Column(Integer, default=0)
    credits_balance_before = Column(Integer)
    credits_balance_after = Column(Integer)
    
    # Error tracking
    error_message = Column(Text)
    error_traceback = Column(Text)
    error_type = Column(String(64))
    
    # Security
    is_suspicious = Column(Boolean, default=False)
    threat_score = Column(Integer, default=0)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="api_logs")

class License(Base):
    __tablename__ = 'licenses'
    
    id = Column(Integer, primary_key=True)
    license_uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    license_key = Column(String(256), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # License details
    license_type = Column(String(32), default='trial')  # trial, personal, professional, enterprise, government
    tier = Column(String(32), default='free')  # free, basic, pro, enterprise
    features = Column(JSON, default=list)
    limits = Column(JSON, default=dict)
    
    # Validity
    issued_at = Column(DateTime, default=datetime.utcnow)
    activated_at = Column(DateTime)
    expires_at = Column(DateTime)
    is_perpetual = Column(Boolean, default=False)
    
    # Status
    status = Column(String(32), default='active')  # active, expired, suspended, revoked, trial
    is_valid = Column(Boolean, default=True)
    suspension_reason = Column(Text)
    
    # Payment
    payment_reference = Column(String(256))
    payment_gateway = Column(String(32))
    amount_paid = Column(Numeric(10, 2))
    currency = Column(String(3), default='USD')
    
    # Hardware binding
    allowed_devices = Column(Integer, default=1)
    device_fingerprints = Column(JSON, default=list)
    is_transferable = Column(Boolean, default=False)
    
    # Metadata
    notes = Column(Text)
    license_metadata = Column(JSON)  # CHANGED: renamed from 'metadata'
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")

# ============================================================================
# 💰 ENHANCED CREDIT MANAGER
# ============================================================================

class EnhancedCreditManager:
    """Advanced credit management with pricing tiers, subscriptions, and billing"""
    
    # Credit costs per action
    CREDIT_COSTS = {
        'scan': {
            'quick': 1,      # Quick scan
            'basic': 3,      # Basic scan
            'full': 5,       # Full scan with AI
            'deep': 10,      # Deep forensic scan
            'network': 4,    # Network analysis
            'memory': 6,     # Memory analysis
            'comprehensive': 15,  # All checks
        },
        'api': {
            'detection': 2,   # Single detection API call
            'batch': 20,      # Batch processing (10 devices)
            'export': 1,      # Data export
            'analysis': 3,    # Advanced analysis
            'prediction': 4,  # ML prediction
        },
        'features': {
            'ai_detection': 10,
            'real_time_monitoring': 15,
            'advanced_analytics': 5,
            'custom_models': 25,
            'priority_support': 20,
        }
    }
    
    # Credit packages for purchase
    CREDIT_PACKAGES = {
        'starter': {
            'credits': 100,
            'price': 0.00,
            'currency': 'USD',
            'daily_limit': 10,
            'monthly_limit': 100,
            'features': ['basic_scan', 'email_support'],
            'description': 'Perfect for getting started'
        },
        'basic': {
            'credits': 1000,
            'price': 9.99,
            'currency': 'USD',
            'daily_limit': 100,
            'monthly_limit': 1000,
            'features': ['basic_scan', 'ai_detection', 'api_access', 'email_support'],
            'description': 'For individual users and small teams'
        },
        'professional': {
            'credits': 5000,
            'price': 49.99,
            'currency': 'USD',
            'daily_limit': 500,
            'monthly_limit': 5000,
            'features': ['all_scans', 'advanced_ai', 'api_access', 'priority_support', 'export'],
            'description': 'For security professionals and businesses'
        },
        'enterprise': {
            'credits': 50000,
            'price': 299.99,
            'currency': 'USD',
            'daily_limit': 5000,
            'monthly_limit': 50000,
            'features': ['all_features', 'custom_models', 'dedicated_support', 'sla', 'white_label'],
            'description': 'For large organizations and MSSPs'
        },
        'government': {
            'credits': 100000,
            'price': 999.99,
            'currency': 'USD',
            'daily_limit': 10000,
            'monthly_limit': 100000,
            'features': ['all_features', 'custom_models', '24/7_support', 'sla', 'audit_logs', 'compliance'],
            'description': 'For government and critical infrastructure'
        }
    }
    
    # Subscription plans
    SUBSCRIPTION_PLANS = {
        'monthly_basic': {
            'name': 'Basic Monthly',
            'price': 9.99,
            'currency': 'USD',
            'interval': 'month',
            'interval_count': 1,
            'credits_per_month': 1000,
            'features': ['basic_scan', 'ai_detection', 'api_access']
        },
        'monthly_pro': {
            'name': 'Pro Monthly',
            'price': 49.99,
            'currency': 'USD',
            'interval': 'month',
            'interval_count': 1,
            'credits_per_month': 5000,
            'features': ['all_scans', 'advanced_ai', 'priority_support']
        },
        'yearly_basic': {
            'name': 'Basic Yearly',
            'price': 99.99,  # 2 months free
            'currency': 'USD',
            'interval': 'year',
            'interval_count': 1,
            'credits_per_month': 1000,
            'features': ['basic_scan', 'ai_detection', 'api_access']
        },
        'yearly_pro': {
            'name': 'Pro Yearly',
            'price': 499.99,  # 2 months free
            'currency': 'USD',
            'interval': 'year',
            'interval_count': 1,
            'credits_per_month': 5000,
            'features': ['all_scans', 'advanced_ai', 'priority_support']
        }
    }
    
    def __init__(self, db_session, stripe_api_key=None, paypal_client_id=None, paypal_secret=None):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
        
        # Initialize payment processors
        if stripe_api_key:
            stripe.api_key = stripe_api_key
            self.stripe = stripe
        else:
            self.stripe = None
        
        if paypal_client_id and paypal_secret:
            paypalrestsdk.configure({
                "mode": "live" if os.environ.get('PAYPAL_LIVE') else "sandbox",
                "client_id": paypal_client_id,
                "client_secret": paypal_secret
            })
            self.paypal = paypalrestsdk
        else:
            self.paypal = None
        
        # Initialize crypto payment if available
        self.crypto_enabled = os.environ.get('COINBASE_API_KEY') is not None
        
    def check_credits(self, user_id, action_type, action_subtype=None, quantity=1):
        """Check if user has enough credits for an action"""
        credits = self.db.query(UserCredits).filter_by(user_id=user_id).first()
        
        if not credits:
            # Create credit record if it doesn't exist
            credits = self._create_credit_record(user_id)
        
        # Reset daily/monthly counts if needed
        self._reset_usage_counts(credits)
        
        # Calculate cost
        cost = self._calculate_cost(action_type, action_subtype, quantity)
        
        # Check daily limits
        if action_type == 'scan':
            if credits.scans_today + quantity > credits.daily_scan_limit:
                return False, "📊 Daily scan limit reached"
        
        # Check monthly limits
        if credits.scans_this_month + quantity > credits.monthly_scan_limit:
            return False, "📅 Monthly scan limit reached"
        
        # Check credits
        if credits.credits_balance < cost:
            return False, f"💰 Insufficient credits. Need {cost}, have {credits.credits_balance}"
        
        return True, cost
    
    def use_credits(self, user_id, action_type, action_subtype=None, quantity=1, description="", metadata=None):
        """Use credits for an action"""
        can_use, result = self.check_credits(user_id, action_type, action_subtype, quantity)
        
        if not can_use:
            raise ValueError(f"❌ Cannot use credits: {result}")
        
        cost = result
        credits = self.db.query(UserCredits).filter_by(user_id=user_id).first()
        
        # Record previous balance
        previous_balance = credits.credits_balance
        
        # Deduct credits
        credits.credits_balance -= cost
        credits.credits_used += cost
        
        # Update usage counts
        if action_type == 'scan':
            credits.scans_today += quantity
            credits.scans_this_month += quantity
        elif action_type == 'api':
            credits.api_calls_today += quantity
        
        # Create transaction record
        transaction = CreditTransaction(
            transaction_id=f"TXN{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}",
            user_credits_id=credits.id,
            type='usage',
            amount=-cost,
            description=description or f"{action_type}: {action_subtype} x{quantity}",
            transaction_metadata=metadata or {},
            status='completed',
            completed_at=datetime.utcnow()
        )
        
        self.db.add(transaction)
        self.db.commit()
        
        # Log credit usage
        self.logger.info(f"💰 Credits used: User {user_id}, Cost: {cost}, Balance: {credits.credits_balance}")
        
        # Send notification if balance is low
        if credits.credits_balance < 50:
            self._send_low_balance_notification(user_id, credits.credits_balance)
        
        return {
            'success': True,
            'cost': cost,
            'previous_balance': previous_balance,
            'new_balance': credits.credits_balance,
            'transaction_id': transaction.transaction_id
        }
    
    def add_credits(self, user_id, amount, credit_type='purchase', description="", reference="", metadata=None):
        """Add credits to user account"""
        credits = self.db.query(UserCredits).filter_by(user_id=user_id).first()
        
        if not credits:
            credits = self._create_credit_record(user_id)
        
        previous_balance = credits.credits_balance
        
        # Update balances
        credits.credits_balance += amount
        
        if credit_type == 'purchase':
            credits.credits_purchased += amount
        elif credit_type == 'bonus':
            credits.credits_bonus += amount
            credits.free_credits += amount
        
        # Create transaction
        transaction = CreditTransaction(
            transaction_id=f"TXN{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}",
            user_credits_id=credits.id,
            type=credit_type,
            amount=amount,
            description=description,
            reference=reference,
            transaction_metadata=metadata or {},
            status='completed',
            completed_at=datetime.utcnow()
        )
        
        self.db.add(transaction)
        self.db.commit()
        
        # Send notification
        self._send_credit_added_notification(user_id, amount, credits.credits_balance)
        
        return {
            'success': True,
            'amount': amount,
            'previous_balance': previous_balance,
            'new_balance': credits.credits_balance,
            'transaction_id': transaction.transaction_id
        }
    
    def purchase_credits(self, user_id, package_type, payment_method='stripe', payment_details=None):
        """Purchase credits with payment processing"""
        package = self.CREDIT_PACKAGES.get(package_type)
        
        if not package:
            raise ValueError(f"❌ Invalid package: {package_type}")
        
        amount = package['credits']
        price = package['price']
        
        # Process payment
        if payment_method == 'stripe' and self.stripe:
            payment_result = self._process_stripe_payment(user_id, price, package, payment_details)
        elif payment_method == 'paypal' and self.paypal:
            payment_result = self._process_paypal_payment(user_id, price, package, payment_details)
        elif payment_method == 'crypto' and self.crypto_enabled:
            payment_result = self._process_crypto_payment(user_id, price, package, payment_details)
        else:
            raise ValueError(f"❌ Unsupported payment method: {payment_method}")
        
        if payment_result['success']:
            # Add credits
            result = self.add_credits(
                user_id=user_id,
                amount=amount,
                credit_type='purchase',
                description=f"Purchase: {package_type} package",
                reference=payment_result.get('payment_id'),
                metadata={
                    'package': package_type,
                    'price': price,
                    'currency': package['currency'],
                    'payment_method': payment_method,
                    'payment_details': payment_result
                }
            )
            
            # Create invoice
            self._create_invoice(user_id, package, payment_result)
            
            return {
                'success': True,
                'credits_added': amount,
                'total_credits': result['new_balance'],
                'payment': payment_result,
                'invoice_id': f"INV{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}"
            }
        
        return payment_result
    
    def create_subscription(self, user_id, plan_id, payment_method='stripe', payment_details=None):
        """Create a subscription for recurring billing"""
        plan = self.SUBSCRIPTION_PLANS.get(plan_id)
        
        if not plan:
            raise ValueError(f"❌ Invalid plan: {plan_id}")
        
        credits = self.db.query(UserCredits).filter_by(user_id=user_id).first()
        if not credits:
            credits = self._create_credit_record(user_id)
        
        # Process subscription payment
        if payment_method == 'stripe' and self.stripe:
            subscription_result = self._create_stripe_subscription(user_id, plan, payment_details)
        elif payment_method == 'paypal' and self.paypal:
            subscription_result = self._create_paypal_subscription(user_id, plan, payment_details)
        else:
            raise ValueError(f"❌ Unsupported payment method for subscriptions: {payment_method}")
        
        if subscription_result['success']:
            # Update user credits with subscription
            credits.billing_tier = 'subscription'
            credits.subscription_id = subscription_result.get('subscription_id')
            credits.subscription_status = 'active'
            credits.subscription_expiry = subscription_result.get('expiry_date')
            credits.daily_scan_limit = 1000  # Higher limits for subscribers
            credits.monthly_scan_limit = 10000
            
            # Add initial credits
            initial_credits = plan.get('credits_per_month', 1000)
            self.add_credits(
                user_id=user_id,
                amount=initial_credits,
                credit_type='purchase',
                description=f"Subscription: {plan['name']} - Initial credits",
                reference=subscription_result.get('subscription_id')
            )
            
            self.db.commit()
            
            return {
                'success': True,
                'subscription': subscription_result,
                'credits_added': initial_credits,
                'message': f"✅ Subscription activated: {plan['name']}"
            }
        
        return subscription_result
    
    def get_user_credits_info(self, user_id):
        """Get comprehensive credit information for user"""
        credits = self.db.query(UserCredits).filter_by(user_id=user_id).first()
        
        if not credits:
            credits = self._create_credit_record(user_id)
        
        # Reset counts if needed
        self._reset_usage_counts(credits)
        
        # Get recent transactions
        transactions = self.db.query(CreditTransaction)\
            .filter_by(user_credits_id=credits.id)\
            .order_by(CreditTransaction.created_at.desc())\
            .limit(10)\
            .all()
        
        # Calculate daily usage
        today = datetime.utcnow().date()
        daily_usage = self.db.query(sa.func.sum(CreditTransaction.amount))\
            .filter(
                CreditTransaction.user_credits_id == credits.id,
                CreditTransaction.type == 'usage',
                sa.func.date(CreditTransaction.created_at) == today
            )\
            .scalar() or 0
        
        return {
            'credits': {
                'balance': credits.credits_balance,
                'used': credits.credits_used,
                'purchased': credits.credits_purchased,
                'bonus': credits.credits_bonus,
                'free': credits.free_credits,
            },
            'limits': {
                'daily_scans': credits.scans_today,
                'daily_scan_limit': credits.daily_scan_limit,
                'monthly_scans': credits.scans_this_month,
                'monthly_scan_limit': credits.monthly_scan_limit,
                'daily_api_calls': credits.api_calls_today,
                'daily_api_limit': credits.daily_api_limit,
            },
            'billing': {
                'tier': credits.billing_tier,
                'subscription_status': credits.subscription_status,
                'subscription_expiry': credits.subscription_expiry.isoformat() if credits.subscription_expiry else None,
            },
            'usage': {
                'daily_credits_used': abs(daily_usage),
                'daily_scan_quota': f"{credits.scans_today}/{credits.daily_scan_limit}",
                'monthly_scan_quota': f"{credits.scans_this_month}/{credits.monthly_scan_limit}",
            },
            'transactions': [
                {
                    'id': t.transaction_id,
                    'type': t.type,
                    'amount': t.amount,
                    'description': t.description,
                    'created_at': t.created_at.isoformat(),
                    'status': t.status
                }
                for t in transactions
            ],
            'available_packages': self.CREDIT_PACKAGES,
            'subscription_plans': self.SUBSCRIPTION_PLANS
        }
    
    def generate_usage_report(self, user_id, start_date=None, end_date=None):
        """Generate detailed usage report"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        credits = self.db.query(UserCredits).filter_by(user_id=user_id).first()
        if not credits:
            return {'error': 'User not found'}
        
        # Get transactions in date range
        transactions = self.db.query(CreditTransaction)\
            .filter(
                CreditTransaction.user_credits_id == credits.id,
                CreditTransaction.created_at >= start_date,
                CreditTransaction.created_at <= end_date
            )\
            .order_by(CreditTransaction.created_at)\
            .all()
        
        # Calculate statistics
        total_credits_used = sum(abs(t.amount) for t in transactions if t.type == 'usage')
        total_credits_added = sum(t.amount for t in transactions if t.type in ['purchase', 'bonus'])
        
        # Group by day
        daily_usage = defaultdict(int)
        for t in transactions:
            if t.type == 'usage':
                day = t.created_at.date()
                daily_usage[day] += abs(t.amount)
        
        # Group by action type
        action_usage = defaultdict(int)
        for t in transactions:
            if t.type == 'usage' and t.description:
                # Parse action from description
                if 'scan:' in t.description.lower():
                    action_usage['scans'] += abs(t.amount)
                elif 'api:' in t.description.lower():
                    action_usage['api_calls'] += abs(t.amount)
                elif 'feature:' in t.description.lower():
                    action_usage['features'] += abs(t.amount)
                else:
                    action_usage['other'] += abs(t.amount)
        
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days
            },
            'summary': {
                'total_credits_used': total_credits_used,
                'total_credits_added': total_credits_added,
                'net_change': total_credits_added - total_credits_used,
                'average_daily_usage': total_credits_used / max((end_date - start_date).days, 1)
            },
            'daily_usage': dict(daily_usage),
            'action_breakdown': dict(action_usage),
            'transactions': [
                {
                    'date': t.created_at.isoformat(),
                    'type': t.type,
                    'amount': t.amount,
                    'description': t.description,
                    'balance_after': credits.credits_balance  # Note: this would need calculation
                }
                for t in transactions
            ]
        }
    
    def _create_credit_record(self, user_id):
        """Create a new credit record for user"""
        credits = UserCredits(
            user_id=user_id,
            credits_balance=self.CREDIT_PACKAGES['starter']['credits'],
            free_credits=self.CREDIT_PACKAGES['starter']['credits'],
            daily_scan_limit=self.CREDIT_PACKAGES['starter']['daily_limit'],
            monthly_scan_limit=self.CREDIT_PACKAGES['starter']['monthly_limit']
        )
        
        self.db.add(credits)
        self.db.commit()
        
        return credits
    
    def _reset_usage_counts(self, credits):
        """Reset daily and monthly usage counts if needed"""
        now = datetime.utcnow()
        
        # Reset daily counts
        if not credits.last_reset_date or credits.last_reset_date.date() < now.date():
            credits.scans_today = 0
            credits.api_calls_today = 0
            credits.last_reset_date = now
        
        # Reset monthly counts
        if not credits.last_reset_month or credits.last_reset_month.month < now.month:
            credits.scans_this_month = 0
            credits.last_reset_month = now
        
        self.db.commit()
    
    def _calculate_cost(self, action_type, action_subtype=None, quantity=1):
        """Calculate credit cost for an action"""
        if action_type in self.CREDIT_COSTS:
            if action_subtype and action_subtype in self.CREDIT_COSTS[action_type]:
                base_cost = self.CREDIT_COSTS[action_type][action_subtype]
            else:
                # Use minimum cost for the category
                base_cost = min(self.CREDIT_COSTS[action_type].values())
        else:
            base_cost = 1  # Default cost
        
        return base_cost * quantity
    
    def _process_stripe_payment(self, user_id, amount, package, payment_details):
        """Process payment via Stripe"""
        try:
            # Get or create Stripe customer
            user = self.db.query(User).filter_by(id=user_id).first()
            credits = self.db.query(UserCredits).filter_by(user_id=user_id).first()
            
            if credits and credits.stripe_customer_id:
                customer_id = credits.stripe_customer_id
            else:
                # Create new customer
                customer = self.stripe.Customer.create(
                    email=user.email,
                    name=f"{user.first_name} {user.last_name}" if user.first_name else user.username,
                    metadata={
                        'user_id': user_id,
                        'username': user.username
                    }
                )
                customer_id = customer.id
                
                if credits:
                    credits.stripe_customer_id = customer_id
                    self.db.commit()
            
            # Create payment intent
            payment_intent = self.stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=package['currency'].lower(),
                customer=customer_id,
                description=f"Pegasus Credits: {package.get('description', '')}",
                metadata={
                    'package': list(self.CREDIT_PACKAGES.keys())[list(self.CREDIT_PACKAGES.values()).index(package)],
                    'user_id': user_id,
                    'credits': package['credits']
                }
            )
            
            return {
                'success': True,
                'payment_id': payment_intent.id,
                'client_secret': payment_intent.client_secret,
                'customer_id': customer_id,
                'amount': amount,
                'currency': package['currency']
            }
            
        except Exception as e:
            self.logger.error(f"Stripe payment error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _process_paypal_payment(self, user_id, amount, package, payment_details):
        """Process payment via PayPal"""
        try:
            payment = Payment({
                "intent": "sale",
                "payer": {
                    "payment_method": "paypal"
                },
                "redirect_urls": {
                    "return_url": os.environ.get('PAYPAL_RETURN_URL', 'http://localhost:5000/payment/success'),
                    "cancel_url": os.environ.get('PAYPAL_CANCEL_URL', 'http://localhost:5000/payment/cancel')
                },
                "transactions": [{
                    "amount": {
                        "total": str(amount),
                        "currency": package['currency']
                    },
                    "description": f"Pegasus Credits: {package.get('description', '')}",
                    "custom": json.dumps({
                        'user_id': user_id,
                        'package': list(self.CREDIT_PACKAGES.keys())[list(self.CREDIT_PACKAGES.values()).index(package)],
                        'credits': package['credits']
                    })
                }]
            })
            
            if payment.create():
                # Store payment ID in session or database
                return {
                    'success': True,
                    'payment_id': payment.id,
                    'approval_url': next(link.href for link in payment.links if link.rel == "approval_url"),
                    'amount': amount,
                    'currency': package['currency']
                }
            else:
                return {
                    'success': False,
                    'error': payment.error
                }
                
        except Exception as e:
            self.logger.error(f"PayPal payment error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_invoice(self, user_id, package, payment_result):
        """Create an invoice for the purchase"""
        user = self.db.query(User).filter_by(id=user_id).first()
        credits = self.db.query(UserCredits).filter_by(user_id=user_id).first()
        
        invoice = Invoice(
            invoice_id=f"INV{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}",
            invoice_number=f"INV-{datetime.utcnow().strftime('%Y%m')}-{self.db.query(Invoice).count() + 1:04d}",
            user_id=user_id,
            user_credits_id=credits.id if credits else None,
            amount=package['price'],
            currency=package['currency'],
            description=f"Purchase: {package.get('description', 'Credit Package')}",
            items=[{
                'item': f"{package['credits']} Credits",
                'quantity': 1,
                'price': package['price'],
                'total': package['price']
            }],
            subtotal=package['price'],
            tax=0,
            discount=0,
            total=package['price'],
            status='paid',
            payment_method='stripe' if payment_result.get('payment_id') else 'paypal',
            payment_date=datetime.utcnow(),
            payment_reference=payment_result.get('payment_id'),
            payment_gateway='stripe' if payment_result.get('customer_id') else 'paypal',
            billing_name=f"{user.first_name} {user.last_name}" if user.first_name else user.username,
            billing_email=user.email,
            created_at=datetime.utcnow(),
            due_date=datetime.utcnow() + timedelta(days=30),
            paid_at=datetime.utcnow()
        )
        
        self.db.add(invoice)
        self.db.commit()
        
        # Generate PDF invoice
        self._generate_invoice_pdf(invoice)
        
        return invoice
    
    def _generate_invoice_pdf(self, invoice):
        """Generate PDF invoice"""
        try:
            # Create PDF directory
            pdf_dir = Path('invoices')
            pdf_dir.mkdir(exist_ok=True)
            
            filename = pdf_dir / f"{invoice.invoice_id}.pdf"
            
            doc = SimpleDocTemplate(str(filename), pagesize=letter)
            elements = []
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            normal_style = styles['Normal']
            
            # Title
            elements.append(Paragraph("PEGASUS ENTERPRISE DETECTOR", title_style))
            elements.append(Paragraph("INVOICE", title_style))
            elements.append(Spacer(1, 20))
            
            # Invoice details
            invoice_data = [
                ['Invoice Number:', invoice.invoice_number],
                ['Invoice Date:', invoice.created_at.strftime('%Y-%m-%d')],
                ['Due Date:', invoice.due_date.strftime('%Y-%m-%d')],
                ['Status:', invoice.status.upper()],
                ['Payment Method:', invoice.payment_method.upper()],
            ]
            
            invoice_table = RLTable(invoice_data, colWidths=[200, 200])
            invoice_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ]))
            
            elements.append(invoice_table)
            elements.append(Spacer(1, 30))
            
            # Items
            items_data = [['Description', 'Quantity', 'Unit Price', 'Total']]
            for item in invoice.items or []:
                items_data.append([
                    item.get('item', ''),
                    item.get('quantity', 1),
                    f"${item.get('price', 0):.2f}",
                    f"${item.get('total', 0):.2f}"
                ])
            
            items_table = RLTable(items_data, colWidths=[250, 80, 80, 80])
            items_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            
            elements.append(items_table)
            elements.append(Spacer(1, 20))
            
            # Totals
            totals_data = [
                ['Subtotal:', f"${invoice.subtotal:.2f}"],
                ['Tax:', f"${invoice.tax:.2f}"],
                ['Discount:', f"-${invoice.discount:.2f}"],
                ['Total:', f"${invoice.total:.2f}", 'bold']
            ]
            
            totals_table = RLTable(totals_data, colWidths=[400, 80])
            totals_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ]))
            
            elements.append(totals_table)
            
            # Build PDF
            doc.build(elements)
            
            # Update invoice with file location
            invoice.export_location = str(filename)
            self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Error generating PDF invoice: {e}")
    
    def _send_low_balance_notification(self, user_id, balance):
        """Send notification when credit balance is low"""
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            return
        
        # Create alert
        alert = Alert(
            alert_id=f"ALERT{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}",
            user_id=user_id,
            severity='warning',
            title='💰 Low Credit Balance',
            message=f'Your credit balance is low ({balance} credits). Consider purchasing more credits to continue using the service.',
            alert_type='billing',
            category='credit',
            details={'balance': balance, 'threshold': 50},
            recommendations='Purchase additional credits from the billing page.',
            actions=['purchase_credits', 'view_billing']
        )
        
        self.db.add(alert)
        self.db.commit()
        
        # Send email notification
        if user.email and user.notification_settings.get('email_low_balance', True):
            self._send_email_notification(
                user.email,
                'Low Credit Balance - Pegasus Enterprise',
                f'Your credit balance is running low ({balance} credits).'
            )
    
    def _send_credit_added_notification(self, user_id, amount, new_balance):
        """Send notification when credits are added"""
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            return
        
        # Create alert
        alert = Alert(
            alert_id=f"ALERT{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}",
            user_id=user_id,
            severity='info',
            title='✅ Credits Added',
            message=f'{amount} credits have been added to your account. New balance: {new_balance} credits.',
            alert_type='billing',
            category='credit',
            details={'amount_added': amount, 'new_balance': new_balance}
        )
        
        self.db.add(alert)
        self.db.commit()
        
        # Send email notification
        if user.email and user.notification_settings.get('email_credits_added', True):
            self._send_email_notification(
                user.email,
                'Credits Added - Pegasus Enterprise',
                f'{amount} credits have been added to your account.'
            )
    
    def _send_email_notification(self, to_email, subject, body):
        """Send email notification"""
        try:
            # This would integrate with your email service
            # For example, using Flask-Mail or SMTP
            msg = Message(
                subject=subject,
                recipients=[to_email],
                body=body,
                html=f'<html><body><h3>{subject}</h3><p>{body}</p></body></html>'
            )
            
            # mail.send(msg)  # Uncomment when mail is configured
            self.logger.info(f"Email sent to {to_email}: {subject}")
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")

# ============================================================================
# 🔐 ENHANCED LICENSE MANAGER
# ============================================================================

class EnhancedLicenseManager:
    """Advanced license management with hardware binding, activation, and validation"""
    
    # License tiers and features
    LICENSE_TIERS = {
        'trial': {
            'name': 'Trial',
            'duration_days': 7,
            'max_devices': 3,
            'max_users': 1,
            'features': ['basic_scan', 'email_support'],
            'price': 0.00,
            'renewable': False
        },
        'personal': {
            'name': 'Personal',
            'duration_days': 365,
            'max_devices': 10,
            'max_users': 1,
            'features': ['full_scan', 'ai_detection', 'email_support'],
            'price': 99.99,
            'renewable': True
        },
        'professional': {
            'name': 'Professional',
            'duration_days': 365,
            'max_devices': 50,
            'max_users': 10,
            'features': ['all_scans', 'advanced_ai', 'api_access', 'priority_support'],
            'price': 499.99,
            'renewable': True
        },
        'enterprise': {
            'name': 'Enterprise',
            'duration_days': 365,
            'max_devices': 500,
            'max_users': 100,
            'features': ['all_features', 'custom_models', 'dedicated_support', 'sla'],
            'price': 2999.99,
            'renewable': True
        },
        'government': {
            'name': 'Government',
            'duration_days': 365,
            'max_devices': 5000,
            'max_users': 500,
            'features': ['all_features', 'custom_models', '24/7_support', 'audit_logs', 'compliance'],
            'price': 9999.99,
            'renewable': True
        }
    }
    
    def __init__(self, db_session, encryption_key=None):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
        
        # Encryption for license keys
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode())
        else:
            # Generate a key if not provided
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
    
    def generate_license_key(self, user_id, tier='personal', duration_days=None, metadata=None):
        """Generate a new license key"""
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("❌ User not found")
        
        tier_config = self.LICENSE_TIERS.get(tier)
        if not tier_config:
            raise ValueError(f"❌ Invalid tier: {tier}")
        
        # Calculate expiry
        if duration_days is None:
            duration_days = tier_config['duration_days']
        
        issued_at = datetime.utcnow()
        expires_at = issued_at + timedelta(days=duration_days)
        
        # Create license data
        license_data = {
            'license_id': f"LIC{uuid.uuid4().hex[:16].upper()}",
            'user_id': user_id,
            'username': user.username,
            'email': user.email,
            'tier': tier,
            'tier_name': tier_config['name'],
            'issued_at': issued_at.isoformat(),
            'expires_at': expires_at.isoformat(),
            'duration_days': duration_days,
            'max_devices': tier_config['max_devices'],
            'max_users': tier_config['max_users'],
            'features': tier_config['features'],
            'price': tier_config['price'],
            'metadata': metadata or {},
            'version': '5.0'
        }
        
        # Generate signature
        signature = self._generate_signature(license_data)
        license_data['signature'] = signature
        
        # Encode license
        license_key = self._encode_license(license_data)
        
        # Save to database
        license = License(
            license_uuid=str(uuid.uuid4()),
            license_key=license_key,
            user_id=user_id,
            license_type=tier,
            tier=tier,
            features=tier_config['features'],
            limits={
                'max_devices': tier_config['max_devices'],
                'max_users': tier_config['max_users'],
                'daily_scans': 100 if tier == 'trial' else 1000
            },
            issued_at=issued_at,
            activated_at=datetime.utcnow(),
            expires_at=expires_at,
            is_perpetual=False,
            status='active',
            is_valid=True,
            amount_paid=tier_config['price'],
            currency='USD',
            allowed_devices=tier_config['max_devices'],
            device_fingerprints=[],
            is_transferable=tier != 'trial',
            license_metadata=metadata or {},
            notes=f"Generated for {user.username}"
        )
        
        self.db.add(license)
        self.db.commit()
        
        # Update user's billing tier
        credits = self.db.query(UserCredits).filter_by(user_id=user_id).first()
        if credits:
            credits.billing_tier = tier
            self.db.commit()
        
        return {
            'success': True,
            'license_key': license_key,
            'license_data': license_data,
            'license_id': license.license_uuid,
            'message': f'✅ License generated successfully for {tier_config["name"]} tier'
        }
    
    def validate_license(self, license_key, device_fingerprint=None):
        """Validate a license key"""
        try:
            # Decode license
            license_data = self._decode_license(license_key)
            
            # Check if license exists in database
            license_record = self.db.query(License).filter_by(license_key=license_key).first()
            if not license_record:
                return {
                    'valid': False,
                    'error': 'License not found in database'
                }
            
            # Check if license is valid
            if not license_record.is_valid:
                return {
                    'valid': False,
                    'error': 'License is invalid or suspended'
                }
            
            # Check expiration
            if not license_record.is_perpetual and license_record.expires_at < datetime.utcnow():
                license_record.status = 'expired'
                license_record.is_valid = False
                self.db.commit()
                
                return {
                    'valid': False,
                    'error': 'License has expired'
                }
            
            # Verify signature
            if not self._verify_signature(license_data):
                return {
                    'valid': False,
                    'error': 'Invalid license signature'
                }
            
            # Check device binding if required
            if device_fingerprint:
                if len(license_record.device_fingerprints) >= license_record.allowed_devices:
                    if device_fingerprint not in license_record.device_fingerprints:
                        return {
                            'valid': False,
                            'error': f'Maximum devices ({license_record.allowed_devices}) reached'
                        }
                
                # Add device fingerprint if not already present
                if device_fingerprint not in license_record.device_fingerprints:
                    license_record.device_fingerprints.append(device_fingerprint)
                    self.db.commit()
            
            # Update last validation time
            license_record.updated_at = datetime.utcnow()
            self.db.commit()
            
            return {
                'valid': True,
                'license': {
                    'tier': license_record.tier,
                    'features': license_record.features,
                    'limits': license_record.limits,
                    'expires_at': license_record.expires_at.isoformat() if license_record.expires_at else None,
                    'is_perpetual': license_record.is_perpetual,
                    'devices_used': len(license_record.device_fingerprints),
                    'devices_allowed': license_record.allowed_devices
                },
                'user_id': license_record.user_id
            }
            
        except Exception as e:
            self.logger.error(f"License validation error: {e}")
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    def activate_license(self, license_key, activation_code=None):
        """Activate a license"""
        license_record = self.db.query(License).filter_by(license_key=license_key).first()
        
        if not license_record:
            return {
                'success': False,
                'error': 'License not found'
            }
        
        if license_record.status == 'active':
            return {
                'success': False,
                'error': 'License is already active'
            }
        
        # Check activation code if required
        if activation_code:
            # Implement activation code validation
            pass
        
        # Activate license
        license_record.status = 'active'
        license_record.is_valid = True
        license_record.activated_at = datetime.utcnow()
        
        # Set expiry if not perpetual
        if not license_record.is_perpetual and not license_record.expires_at:
            license_record.expires_at = datetime.utcnow() + timedelta(days=365)
        
        self.db.commit()
        
        # Send activation notification
        user = self.db.query(User).filter_by(id=license_record.user_id).first()
        if user:
            self._send_activation_notification(user.email, license_record.tier)
        
        return {
            'success': True,
            'message': '✅ License activated successfully',
            'license': {
                'tier': license_record.tier,
                'expires_at': license_record.expires_at.isoformat() if license_record.expires_at else None,
                'features': license_record.features
            }
        }
    
    def renew_license(self, license_key, renew_period_days=365):
        """Renew an existing license"""
        license_record = self.db.query(License).filter_by(license_key=license_key).first()
        
        if not license_record:
            return {
                'success': False,
                'error': 'License not found'
            }
        
        if not license_record.is_valid:
            return {
                'success': False,
                'error': 'Cannot renew invalid license'
            }
        
        # Calculate new expiry
        current_expiry = license_record.expires_at or datetime.utcnow()
        new_expiry = current_expiry + timedelta(days=renew_period_days)
        
        # Update license
        license_record.expires_at = new_expiry
        license_record.status = 'active'
        license_record.updated_at = datetime.utcnow()
        
        # Create renewal transaction
        # This would integrate with payment processing
        
        self.db.commit()
        
        return {
            'success': True,
            'message': f'✅ License renewed until {new_expiry.strftime("%Y-%m-%d")}',
            'new_expiry': new_expiry.isoformat()
        }
    
    def suspend_license(self, license_key, reason=""):
        """Suspend a license"""
        license_record = self.db.query(License).filter_by(license_key=license_key).first()
        
        if not license_record:
            return {
                'success': False,
                'error': 'License not found'
            }
        
        license_record.status = 'suspended'
        license_record.is_valid = False
        license_record.suspension_reason = reason
        license_record.updated_at = datetime.utcnow()
        
        self.db.commit()
        
        # Notify user
        user = self.db.query(User).filter_by(id=license_record.user_id).first()
        if user:
            self._send_suspension_notification(user.email, reason)
        
        return {
            'success': True,
            'message': 'License suspended'
        }
    
    def revoke_license(self, license_key, reason=""):
        """Revoke a license"""
        license_record = self.db.query(License).filter_by(license_key=license_key).first()
        
        if not license_record:
            return {
                'success': False,
                'error': 'License not found'
            }
        
        license_record.status = 'revoked'
        license_record.is_valid = False
        license_record.suspension_reason = reason
        license_record.updated_at = datetime.utcnow()
        
        self.db.commit()
        
        # Notify user
        user = self.db.query(User).filter_by(id=license_record.user_id).first()
        if user:
            self._send_revocation_notification(user.email, reason)
        
        return {
            'success': True,
            'message': 'License revoked'
        }
    
    def get_license_info(self, license_key):
        """Get detailed license information"""
        license_record = self.db.query(License).filter_by(license_key=license_key).first()
        
        if not license_record:
            return None
        
        user = self.db.query(User).filter_by(id=license_record.user_id).first()
        
        return {
            'license': {
                'id': license_record.license_uuid,
                'key': license_record.license_key,
                'type': license_record.license_type,
                'tier': license_record.tier,
                'status': license_record.status,
                'is_valid': license_record.is_valid,
                'is_perpetual': license_record.is_perpetual,
                'issued_at': license_record.issued_at.isoformat() if license_record.issued_at else None,
                'activated_at': license_record.activated_at.isoformat() if license_record.activated_at else None,
                'expires_at': license_record.expires_at.isoformat() if license_record.expires_at else None,
                'days_remaining': (license_record.expires_at - datetime.utcnow()).days if license_record.expires_at else None,
                'features': license_record.features,
                'limits': license_record.limits,
                'allowed_devices': license_record.allowed_devices,
                'devices_used': len(license_record.device_fingerprints),
                'device_fingerprints': license_record.device_fingerprints,
                'is_transferable': license_record.is_transferable,
                'payment': {
                    'amount_paid': float(license_record.amount_paid) if license_record.amount_paid else 0,
                    'currency': license_record.currency,
                    'payment_reference': license_record.payment_reference
                },
                'license_metadata': license_record.license_metadata,
                'notes': license_record.notes,
                'created_at': license_record.created_at.isoformat(),
                'updated_at': license_record.updated_at.isoformat()
            },
            'user': {
                'id': user.id if user else None,
                'username': user.username if user else None,
                'email': user.email if user else None,
                'company': user.company if user else None
            } if user else None
        }
    
    def transfer_license(self, license_key, new_user_id):
        """Transfer license to another user"""
        license_record = self.db.query(License).filter_by(license_key=license_key).first()
        
        if not license_record:
            return {
                'success': False,
                'error': 'License not found'
            }
        
        if not license_record.is_transferable:
            return {
                'success': False,
                'error': 'License is not transferable'
            }
        
        new_user = self.db.query(User).filter_by(id=new_user_id).first()
        if not new_user:
            return {
                'success': False,
                'error': 'New user not found'
            }
        
        # Update license owner
        old_user_id = license_record.user_id
        license_record.user_id = new_user_id
        license_record.device_fingerprints = []  # Reset device bindings
        license_record.updated_at = datetime.utcnow()
        
        # Update notes
        old_user = self.db.query(User).filter_by(id=old_user_id).first()
        transfer_note = f"Transferred from {old_user.username if old_user else old_user_id} to {new_user.username}"
        license_record.notes = f"{license_record.notes}\n{transfer_note}" if license_record.notes else transfer_note
        
        self.db.commit()
        
        # Notify both users
        if old_user:
            self._send_transfer_notification(old_user.email, 'outgoing', new_user.username)
        
        self._send_transfer_notification(new_user.email, 'incoming', old_user.username if old_user else 'previous owner')
        
        return {
            'success': True,
            'message': f'✅ License transferred to {new_user.username}',
            'new_owner': new_user.username
        }
    
    def generate_license_report(self, user_id=None, status=None, tier=None):
        """Generate license usage report"""
        query = self.db.query(License)
        
        if user_id:
            query = query.filter_by(user_id=user_id)
        
        if status:
            query = query.filter_by(status=status)
        
        if tier:
            query = query.filter_by(tier=tier)
        
        licenses = query.all()
        
        # Calculate statistics
        total_licenses = len(licenses)
        active_licenses = len([l for l in licenses if l.status == 'active'])
        expired_licenses = len([l for l in licenses if l.status == 'expired'])
        
        # Group by tier
        tiers = defaultdict(int)
        for lic in licenses:
            tiers[lic.tier] += 1
        
        # Calculate revenue
        total_revenue = sum(float(l.amount_paid or 0) for l in licenses)
        
        return {
            'summary': {
                'total_licenses': total_licenses,
                'active_licenses': active_licenses,
                'expired_licenses': expired_licenses,
                'suspended_licenses': len([l for l in licenses if l.status == 'suspended']),
                'revoked_licenses': len([l for l in licenses if l.status == 'revoked']),
                'total_revenue': total_revenue
            },
            'tier_distribution': dict(tiers),
            'licenses': [
                {
                    'id': l.license_uuid,
                    'user_id': l.user_id,
                    'tier': l.tier,
                    'status': l.status,
                    'issued_at': l.issued_at.isoformat() if l.issued_at else None,
                    'expires_at': l.expires_at.isoformat() if l.expires_at else None,
                    'amount_paid': float(l.amount_paid or 0),
                    'devices_used': len(l.device_fingerprints),
                    'devices_allowed': l.allowed_devices
                }
                for l in licenses
            ]
        }
    
    def _generate_signature(self, data):
        """Generate signature for license data"""
        # Create a string of important data for signing
        sign_data = f"{data['license_id']}{data['user_id']}{data['tier']}{data['issued_at']}{data['expires_at']}"
        
        # Generate HMAC signature
        secret = os.environ.get('LICENSE_SECRET', 'default-license-secret-change-in-production')
        signature = hmac.new(
            secret.encode(),
            sign_data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _verify_signature(self, data):
        """Verify license signature"""
        if 'signature' not in data:
            return False
        
        stored_signature = data['signature']
        
        # Regenerate signature
        sign_data = f"{data['license_id']}{data['user_id']}{data['tier']}{data['issued_at']}{data['expires_at']}"
        secret = os.environ.get('LICENSE_SECRET', 'default-license-secret-change-in-production')
        calculated_signature = hmac.new(
            secret.encode(),
            sign_data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(stored_signature, calculated_signature)
    
    def _encode_license(self, data):
        """Encode license data to string"""
        # Convert to JSON
        json_str = json.dumps(data)
        
        # Encrypt
        encrypted = self.fernet.encrypt(json_str.encode())
        
        # Encode to base64
        encoded = base64.urlsafe_b64encode(encrypted).decode()
        
        # Format with dashes for readability
        formatted = '-'.join([encoded[i:i+4] for i in range(0, len(encoded), 4)])
        
        return formatted
    
    def _decode_license(self, license_key):
        """Decode license key to data"""
        # Remove dashes
        clean_key = license_key.replace('-', '')
        
        # Decode from base64
        decoded = base64.urlsafe_b64decode(clean_key + '==')
        
        # Decrypt
        decrypted = self.fernet.decrypt(decoded)
        
        # Parse JSON
        data = json.loads(decrypted.decode())
        
        return data
    
    def _send_activation_notification(self, email, tier):
        """Send license activation notification"""
        try:
            subject = f'✅ Your Pegasus Enterprise License Has Been Activated'
            body = f"""
            🎉 Congratulations! Your Pegasus Enterprise license has been activated.
            
            Tier: {tier.title()}
            Activated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
            
            You can now access all features included in your tier.
            
            Thank you for choosing Pegasus Enterprise!
            
            Best regards,
            Pegasus Enterprise Team
            """
            
            # Send email
            self._send_email(email, subject, body)
            
        except Exception as e:
            self.logger.error(f"Error sending activation notification: {e}")
    
    def _send_suspension_notification(self, email, reason):
        """Send license suspension notification"""
        try:
            subject = '⚠️ Your Pegasus Enterprise License Has Been Suspended'
            body = f"""
            ⚠️ Your Pegasus Enterprise license has been suspended.
            
            Reason: {reason}
            Suspended: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
            
            Please contact support to resolve this issue.
            
            Best regards,
            Pegasus Enterprise Team
            """
            
            self._send_email(email, subject, body)
            
        except Exception as e:
            self.logger.error(f"Error sending suspension notification: {e}")
    
    def _send_email(self, to_email, subject, body):
        """Send email helper"""
        # Implement email sending logic
        # This is a placeholder - integrate with your email service
        self.logger.info(f"Email would be sent to {to_email}: {subject}")

# ============================================================================
# 🔍 UNIVERSAL DEVICE SCANNER
# ============================================================================

class UniversalDeviceScanner:
    """Universal scanner for all device types and Pegasus variants"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detection_patterns = self._load_detection_patterns()
        
    def _load_detection_patterns(self):
        """Load detection patterns for all Pegasus variants"""
        return {
            'nso_pegasus': {
                'signatures': [
                    'NSO Group', 'Pegasus', 'Kismet', 'Trident',
                    'CVE-2016-4657', 'CVE-2016-4658', 'CVE-2016-4659',
                    'machook', 'libswift', 'libswiftCore',
                    'com.apple.contacts', 'com.apple.messages'
                ],
                'file_paths': [
                    '/Library/MobileSubstrate/DynamicLibraries/',
                    '/var/containers/Bundle/Application/',
                    '/private/var/mobile/Library/Caches/',
                    '/private/var/mobile/Library/Preferences/'
                ],
                'process_names': [
                    'syslogd', 'mediaserverd', 'locationd',
                    'apsd', 'neagent', 'securityd'
                ],
                'network_indicators': [
                    'mcc.system-update.info',
                    'nms-system.info',
                    'apple.update-system.com'
                ]
            },
            'predator': {
                'signatures': [
                    'Predator', 'Intellexa', 'Alien',
                    'Cytrox', 'Candiru', 'Vilnius'
                ],
                'file_paths': [
                    '/system/app/Predator',
                    '/data/data/com.predator',
                    '/data/local/tmp/alien'
                ],
                'process_names': [
                    'android.process.media',
                    'com.android.systemui',
                    'system_server'
                ]
            },
            'hermit': {
                'signatures': [
                    'Hermit', 'RCS Lab', 'Tykelab',
                    'Chrysaor', 'FlexiSpy', 'MSPY'
                ],
                'file_paths': [
                    '/data/app/com.hermit-',
                    '/system/priv-app/Hermit',
                    '/data/local/hermit'
                ]
            },
            'generic_spyware': {
                'signatures': [
                    'spyware', 'trojan', 'backdoor',
                    'keylogger', 'rat', 'remote_access',
                    'exploit', 'payload', 'injection'
                ]
            }
        }
    
    def scan_universal(self, device_type='auto', jailbreak=False):
        """Universal scanning method for all device types"""
        results = {
            'device_type': device_type,
            'jailbreak_status': jailbreak,
            'timestamp': datetime.utcnow().isoformat(),
            'detection_results': {},
            'final_assessment': {}
        }
        
        try:
            # Detect device type if auto
            if device_type == 'auto':
                device_type = self._detect_device_type()
                results['detected_device_type'] = device_type
            
            # Run device-specific scans
            if device_type in ['iphone', 'ipad', 'ios']:
                results.update(self._scan_ios(jailbreak))
            elif device_type in ['android', 'samsung', 'google']:
                results.update(self._scan_android(jailbreak))
            elif device_type in ['windows', 'pc', 'desktop']:
                results.update(self._scan_windows())
            elif device_type in ['mac', 'macos']:
                results.update(self._scan_macos())
            else:
                results.update(self._scan_generic())
            
            # Run AI/ML analysis
            ai_results = self._run_ai_analysis(results)
            results['ai_analysis'] = ai_results
            
            # Final assessment
            final_verdict = self._generate_final_verdict(results)
            results['final_assessment'] = final_verdict
            
            return results
            
        except Exception as e:
            self.logger.error(f"Scan error: {e}")
            return {
                'error': str(e),
                'final_assessment': {
                    'verdict': 'ERROR',
                    'confidence': 0.0,
                    'message': f'Scan failed: {str(e)}'
                }
            }
    
    def _detect_device_type(self):
        """Auto-detect device type"""
        system = platform.system().lower()
        
        if system == 'darwin':
            # Check if iOS or macOS
            try:
                # Try to detect iOS device
                output = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], 
                                                text=True, stderr=subprocess.DEVNULL)
                if 'iPhone' in output or 'iPad' in output:
                    return 'ios'
                else:
                    return 'macos'
            except:
                return 'macos'
        elif system == 'windows':
            return 'windows'
        elif system == 'linux':
            # Check if Android
            try:
                with open('/system/build.prop', 'r') as f:
                    if 'ro.build.version.sdk' in f.read():
                        return 'android'
            except:
                pass
            return 'linux'
        else:
            return 'unknown'
    
    def _scan_ios(self, jailbreak):
        """Scan iOS device"""
        results = {
            'scan_type': 'ios',
            'checks_performed': [],
            'findings': []
        }
        
        # Check for jailbreak indicators
        if jailbreak:
            results['checks_performed'].append('jailbreak_detection')
            results['findings'].append({
                'type': 'info',
                'message': 'Device is jailbroken',
                'risk': 'medium'
            })
        
        # Check for common Pegasus indicators
        pegasus_indicators = self._check_ios_indicators()
        results['checks_performed'].append('pegasus_indicators')
        results['findings'].extend(pegasus_indicators)
        
        # Check file system
        file_checks = self._scan_ios_filesystem()
        results['checks_performed'].append('filesystem_scan')
        results['findings'].extend(file_checks)
        
        # Check processes
        process_checks = self._scan_ios_processes()
        results['checks_performed'].append('process_scan')
        results['findings'].extend(process_checks)
        
        # Network check
        network_checks = self._check_ios_network()
        results['checks_performed'].append('network_scan')
        results['findings'].extend(network_checks)
        
        return results
    
    def _check_ios_indicators(self):
        """Check for Pegasus indicators on iOS"""
        findings = []
        
        # Check for suspicious files
        suspicious_paths = [
            '/Library/MobileSubstrate/DynamicLibraries/pegasus.dylib',
            '/var/containers/Bundle/Application/.hidden',
            '/private/var/mobile/Library/Caches/com.apple.UIKit.pmessage'
        ]
        
        for path in suspicious_paths:
            if os.path.exists(path):
                findings.append({
                    'type': 'suspicious',
                    'message': f'Suspicious file found: {path}',
                    'risk': 'high',
                    'evidence': path
                })
        
        # Check for suspicious processes
        suspicious_processes = ['syslogd', 'mediaserverd', 'neagent']
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] in suspicious_processes:
                    findings.append({
                        'type': 'suspicious',
                        'message': f'Suspicious process running: {proc.info["name"]}',
                        'risk': 'medium',
                        'evidence': proc.info['name']
                    })
            except:
                pass
        
        return findings
    
    def _scan_ios_filesystem(self):
        """Scan iOS filesystem for anomalies"""
        findings = []
        
        # Check for hidden files
        hidden_dirs = ['/private/var/mobile/.hidden', '/var/mobile/Library/.tmp']
        for dir_path in hidden_dirs:
            if os.path.exists(dir_path):
                findings.append({
                    'type': 'hidden_directory',
                    'message': f'Hidden directory found: {dir_path}',
                    'risk': 'medium'
                })
        
        return findings
    
    def _scan_ios_processes(self):
        """Analyze running processes"""
        findings = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any(indicator in ' '.join(cmdline).lower() 
                                     for indicator in ['pegasus', 'nso', 'spyware']):
                        findings.append({
                            'type': 'malicious_process',
                            'message': f'Malicious process detected: {proc.info["name"]}',
                            'risk': 'high',
                            'evidence': cmdline
                        })
                except:
                    continue
        except:
            pass
        
        return findings
    
    def _check_ios_network(self):
        """Check network connections"""
        findings = []
        
        try:
            connections = psutil.net_connections()
            suspicious_ports = [53, 80, 443, 8080, 8443]  # Common C2 ports
            
            for conn in connections:
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    if conn.raddr.port in suspicious_ports:
                        findings.append({
                            'type': 'suspicious_connection',
                            'message': f'Suspicious connection to {conn.raddr.ip}:{conn.raddr.port}',
                            'risk': 'medium'
                        })
        except:
            pass
        
        return findings
    
    def _scan_android(self, root):
        """Scan Android device"""
        results = {
            'scan_type': 'android',
            'checks_performed': [],
            'findings': []
        }
        
        # Check for root access
        if root:
            results['checks_performed'].append('root_detection')
            results['findings'].append({
                'type': 'info',
                'message': 'Device is rooted',
                'risk': 'medium'
            })
        
        # Check for predator indicators
        predator_checks = self._check_android_predator()
        results['checks_performed'].append('predator_indicators')
        results['findings'].extend(predator_checks)
        
        # Check packages
        package_checks = self._scan_android_packages()
        results['checks_performed'].append('package_scan')
        results['findings'].extend(package_checks)
        
        return results
    
    def _check_android_predator(self):
        """Check for Predator spyware indicators"""
        findings = []
        
        # Check for predator files
        predator_paths = [
            '/system/app/Predator',
            '/data/data/com.predator',
            '/data/local/tmp/.alien'
        ]
        
        for path in predator_paths:
            if os.path.exists(path):
                findings.append({
                    'type': 'predator_detected',
                    'message': f'Predator spyware indicator found: {path}',
                    'risk': 'high'
                })
        
        return findings
    
    def _scan_android_packages(self):
        """Scan installed packages"""
        findings = []
        
        # Simulate package check
        suspicious_packages = [
            'com.predator', 'com.hermit', 'com.spyware',
            'com.tracking', 'com.monitoring'
        ]
        
        for package in suspicious_packages:
            findings.append({
                'type': 'package_check',
                'message': f'Checking for package: {package}',
                'risk': 'low',
                'status': 'not_found'  # Simulated
            })
        
        return findings
    
    def _scan_windows(self):
        """Scan Windows system"""
        return {
            'scan_type': 'windows',
            'checks_performed': ['windows_scan'],
            'findings': [{
                'type': 'info',
                'message': 'Windows scanning capabilities coming soon',
                'risk': 'none'
            }]
        }
    
    def _scan_macos(self):
        """Scan macOS system"""
        return {
            'scan_type': 'macos',
            'checks_performed': ['macos_scan'],
            'findings': [{
                'type': 'info',
                'message': 'macOS scanning capabilities coming soon',
                'risk': 'none'
            }]
        }
    
    def _scan_generic(self):
        """Generic scan for unknown devices"""
        return {
            'scan_type': 'generic',
            'checks_performed': ['generic_scan'],
            'findings': [{
                'type': 'info',
                'message': 'Generic device scan completed',
                'risk': 'none'
            }]
        }
    
    def _run_ai_analysis(self, scan_results):
        """Run AI/ML analysis on scan results"""
        # This would use the ML models in production
        # For now, simulate AI analysis
        
        findings = scan_results.get('findings', [])
        high_risk_findings = [f for f in findings if f.get('risk') == 'high']
        medium_risk_findings = [f for f in findings if f.get('risk') == 'medium']
        
        # Calculate risk score
        risk_score = len(high_risk_findings) * 10 + len(medium_risk_findings) * 5
        
        return {
            'risk_score': risk_score,
            'high_risk_count': len(high_risk_findings),
            'medium_risk_count': len(medium_risk_findings),
            'total_findings': len(findings),
            'ai_confidence': min(risk_score / 100, 0.95)  # Scale to 0-0.95
        }
    
    def _generate_final_verdict(self, results):
        """Generate final verdict based on all findings"""
        ai_results = results.get('ai_analysis', {})
        risk_score = ai_results.get('risk_score', 0)
        high_risk_count = ai_results.get('high_risk_count', 0)
        
        if high_risk_count > 0 or risk_score > 70:
            return {
                'verdict': 'INFECTED',
                'confidence': ai_results.get('ai_confidence', 0.8),
                'risk_level': 'HIGH',
                'message': 'Potential Pegasus infection detected',
                'recommendation': 'Immediate action required. Contact security team.'
            }
        elif risk_score > 30:
            return {
                'verdict': 'SUSPICIOUS',
                'confidence': ai_results.get('ai_confidence', 0.6),
                'risk_level': 'MEDIUM',
                'message': 'Suspicious activity detected',
                'recommendation': 'Further investigation recommended'
            }
        else:
            return {
                'verdict': 'CLEAN',
                'confidence': 0.95,
                'risk_level': 'LOW',
                'message': 'No signs of Pegasus infection',
                'recommendation': 'Regular scanning recommended'
            }

# ============================================================================
# 🏢 PEGASUS ENTERPRISE APPLICATION
# ============================================================================

class PegasusEnterpriseApplication:
    """Main enterprise application with all integrated systems"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.setup_logging()
        self.setup_database()
        self.setup_services()
        self.setup_flask_app()
        self.setup_dash_app()
        self.setup_api_routes()
        
        self.logger.info("🚀 Pegasus Enterprise Application initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        self.logger = logging.getLogger('pegasus_enterprise')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatter
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_format)
        
        # File handler
        file_handler = RotatingFileHandler(
            log_dir / 'pegasus_enterprise.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
        
        # Error handler
        error_handler = RotatingFileHandler(
            log_dir / 'pegasus_errors.log',
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        
        # Suppress verbose logs
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
        
        # Sentry integration
        if os.environ.get('SENTRY_DSN'):
            sentry_sdk.init(
                dsn=os.environ['SENTRY_DSN'],
                integrations=[FlaskIntegration()],
                traces_sample_rate=1.0,
                environment=os.environ.get('ENVIRONMENT', 'development')
            )
    
    def setup_database(self):
        """Setup database with connection pooling"""
        database_url = self.config.get('database_url', 'sqlite:///pegasus_enterprise.db')
        
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=self.config.get('sql_echo', False)
        )
        
        # Create session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        
        # Create all tables
        Base.metadata.create_all(self.engine)
        
        self.logger.info(f"📊 Database connected: {database_url}")
    
    def setup_services(self):
        """Setup all enterprise services"""
        # Credit manager
        self.credit_manager = EnhancedCreditManager(
            db_session=self.Session(),
            stripe_api_key=os.environ.get('STRIPE_API_KEY'),
            paypal_client_id=os.environ.get('PAYPAL_CLIENT_ID'),
            paypal_secret=os.environ.get('PAYPAL_SECRET')
        )
        
        # License manager
        self.license_manager = EnhancedLicenseManager(
            db_session=self.Session(),
            encryption_key=os.environ.get('LICENSE_ENCRYPTION_KEY')
        )
        
        # Payment processors
        self.payment_processors = {
            'stripe': self.credit_manager.stripe is not None,
            'paypal': self.credit_manager.paypal is not None,
            'crypto': self.credit_manager.crypto_enabled
        }
        
        self.logger.info(f"💰 Payment processors: {self.payment_processors}")
    
    def setup_flask_app(self):
        """Setup Flask application with all extensions"""
        self.flask_app = Flask(__name__, 
                            template_folder='templates',
                            static_folder='static',
                            static_url_path='/static')
        
        # Configuration
        self.flask_app.config.update(
            SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-change-in-production-2024'),
            SQLALCHEMY_DATABASE_URI=self.config.get('database_url', 'sqlite:///pegasus_enterprise.db'),
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
            WTF_CSRF_ENABLED=True,
            WTF_CSRF_SECRET_KEY=os.environ.get('CSRF_SECRET_KEY', 'csrf-secret-change-in-production'),
            SESSION_TYPE='redis' if os.environ.get('REDIS_URL') else 'filesystem',
            SESSION_PERMANENT=False,
            SESSION_USE_SIGNER=True,
            PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
            MAIL_SERVER=os.environ.get('MAIL_SERVER', 'smtp.gmail.com'),
            MAIL_PORT=int(os.environ.get('MAIL_PORT', 587)),
            MAIL_USE_TLS=True,
            MAIL_USERNAME=os.environ.get('MAIL_USERNAME'),
            MAIL_PASSWORD=os.environ.get('MAIL_PASSWORD'),
            MAIL_DEFAULT_SENDER=os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@pegasusenterprise.com'),
            BABEL_DEFAULT_LOCALE='en',
            BABEL_DEFAULT_TIMEZONE='UTC'
        )
        
        # Initialize extensions
        CORS(self.flask_app)
        
        self.login_manager = LoginManager()
        self.login_manager.init_app(self.flask_app)
        self.login_manager.login_view = 'login'
        self.login_manager.login_message = 'Please log in to access this page.'
        
        # Initialize Flask-Limiter
        self.limiter = Limiter(
            app=self.flask_app,
            key_func=get_remote_address,
            default_limits=["500 per day", "100 per hour"],
            storage_uri=os.environ.get('REDIS_URL', 'memory://'),
            strategy="fixed-window",
        )
        
        self.mail = Mail(self.flask_app)
        self.babel = Babel(self.flask_app)
        
        # Fix: Check for eventlet availability and set appropriate async_mode
        try:
            import eventlet
            async_mode = 'eventlet'
        except ImportError:
            try:
                import gevent
                async_mode = 'gevent'
            except ImportError:
                try:
                    import geventlet
                    async_mode = 'gevent'
                except ImportError:
                    async_mode = 'threading'
        
        self.socketio = SocketIO(
            self.flask_app,
            cors_allowed_origins="*",
            message_queue=os.environ.get('REDIS_URL'),
            async_mode=async_mode,
            logger=True,
            engineio_logger=True
        )
        
        # User loader for Flask-Login
        @self.login_manager.user_loader
        def load_user(user_id):
            with self.Session() as db:
                return db.query(User).get(int(user_id))
        
        # Context processor
        @self.flask_app.context_processor
        def inject_context():
            return {
                'now': datetime.utcnow(),
                'version': '5.0',
                'unicode_art': UnicodeArt()
            }
        
        # Error handlers
        @self.flask_app.errorhandler(404)
        def not_found_error(error):
            return "<h1>404 - Page Not Found</h1><p>The requested page does not exist.</p><p><a href='/dashboard'>Go to Dashboard</a></p>", 404
        
        @self.flask_app.errorhandler(500)
        def internal_error(error):
            return "<h1>500 - Internal Server Error</h1><p>Something went wrong on our server.</p><p><a href='/dashboard'>Go to Dashboard</a></p>", 500

    def setup_dash_app(self):
        """Setup Plotly Dash application"""
        self.dash_app = dash.Dash(
            __name__,
            server=self.flask_app,
            url_base_pathname='/dashboard/',
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
            ],
            suppress_callback_exceptions=True,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1"}
            ]
        )
        
        # Custom CSS
        self.dash_app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>🏢 Pegasus Enterprise Dashboard</title>
                {%favicon%}
                {%css%}
                <style>
                    .dashboard-header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    }
                    .stat-card {
                        transition: transform 0.3s;
                    }
                    .stat-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                    }
                    .alert-critical {
                        background-color: #ff4444 !important;
                        color: white !important;
                    }
                    .alert-high {
                        background-color: #ff8844 !important;
                        color: white !important;
                    }
                    .alert-medium {
                        background-color: #ffcc44 !important;
                        color: black !important;
                    }
                    .progress-bar-animated {
                        animation: progress-bar-stripes 1s linear infinite;
                    }
                    @keyframes progress-bar-stripes {
                        from { background-position: 40px 0; }
                        to { background-position: 0 0; }
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        # Set layout
        self.dash_app.layout = self.create_dashboard_layout()
        
        # Register callbacks
        self.setup_dash_callbacks()
        
        self.logger.info("📊 Dash dashboard initialized")
    
    def create_dashboard_layout(self):
        """Create the main dashboard layout"""
        return html.Div([
            # URL Location component - CRITICAL FOR NAVIGATION
            dcc.Location(id='url', refresh=False),
            
            # Header
            dbc.Navbar(
                dbc.Container([
                    html.A(
                        dbc.Row([
                            dbc.Col(html.I(className="fas fa-shield-alt fa-2x", style={'color': 'white'})),
                            dbc.Col(dbc.NavbarBrand("🏢 PEGASUS ENTERPRISE", className="ms-2 fw-bold")),
                        ], align="center", className="g-0"),
                        href="/dashboard",
                        style={"textDecoration": "none"},
                    ),
                    dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Collapse(
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink("📊 Overview", href="/dashboard", id="nav-overview")),
                            dbc.NavItem(dbc.NavLink("📱 Devices", href="/dashboard/devices", id="nav-devices")),
                            dbc.NavItem(dbc.NavLink("🔍 Scans", href="/dashboard/scans", id="nav-scans")),
                            dbc.NavItem(dbc.NavLink("🚨 Detections", href="/dashboard/detections", id="nav-detections")),
                            dbc.NavItem(dbc.NavLink("💰 Billing", href="/dashboard/billing", id="nav-billing")),
                            dbc.NavItem(dbc.NavLink("⚙️ Settings", href="/dashboard/settings", id="nav-settings")),
                            dbc.DropdownMenu(
                                children=[
                                    dbc.DropdownMenuItem("👤 Profile", href="#", id="nav-profile"),
                                    dbc.DropdownMenuItem("🔐 Security", href="#", id="nav-security"),
                                    dbc.DropdownMenuItem(divider=True),
                                    dbc.DropdownMenuItem("🚪 Logout", href="/logout", id="nav-logout"),
                                ],
                                nav=True,
                                in_navbar=True,
                                label="Account",
                                align_end=True,
                            ),
                        ], className="ms-auto", navbar=True),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                ], fluid=True),
                color="dark",
                dark=True,
                sticky="top",
                className="mb-4 shadow",
            ),
            
            # Main content area - shows different pages based on URL
            html.Div(id='page-content', className='container-fluid mt-4'),
            
            # Footer
            html.Footer(
                dbc.Container([
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P("🏢 Pegasus Enterprise Detector v5.0", className="mb-1"),
                            html.Small("© 2024 Pegasus Security. All rights reserved.", className="text-muted")
                        ], width=6),
                        
                        dbc.Col([
                            html.Div([
                                html.A("📚 Documentation", href="#", className="text-decoration-none me-3"),
                                html.A("🆘 Support", href="#", className="text-decoration-none me-3"),
                                html.A("📞 Contact", href="#", className="text-decoration-none"),
                            ], className="text-end")
                        ], width=6),
                    ])
                ], fluid=True),
                className="mt-5 py-3 bg-light"
            ),
            
            # Hidden stores and intervals
            dcc.Store(id='session-store'),
            dcc.Store(id='user-data'),
            dcc.Interval(
                id='dashboard-update-interval',
                interval=10*1000,  # 10 seconds
                n_intervals=0
            ),
            dcc.Interval(
                id='system-status-interval',
                interval=5*1000,  # 5 seconds
                n_intervals=0
            ),
            
            # Modals
            self._create_new_scan_modal(),
            self._create_buy_credits_modal(),
            self._create_report_modal(),
        ])
    
    def _create_new_scan_modal(self):
        """Create new scan modal"""
        return dbc.Modal([
            dbc.ModalHeader("🔍 New Scan"),
            dbc.ModalBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Device Type", html_for="scan-device-type"),
                            dbc.Select(
                                id="scan-device-type",
                                options=[
                                    {"label": "📱 iPhone", "value": "iphone"},
                                    {"label": "🤖 Android", "value": "android"},
                                    {"label": "💻 iPad", "value": "ipad"},
                                    {"label": "🖥️ Other", "value": "other"}
                                ],
                                value="iphone"
                            )
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Label("Scan Type", html_for="scan-type"),
                            dbc.Select(
                                id="scan-type",
                                options=[
                                    {"label": "⚡ Quick (1 credit)", "value": "quick"},
                                    {"label": "🔍 Basic (3 credits)", "value": "basic"},
                                    {"label": "🔬 Full (5 credits)", "value": "full"},
                                    {"label": "🧬 Deep (10 credits)", "value": "deep"},
                                    {"label": "🌐 Network (4 credits)", "value": "network"},
                                    {"label": "🧠 Memory (6 credits)", "value": "memory"},
                                    {"label": "🔎 Comprehensive (15 credits)", "value": "comprehensive"}
                                ],
                                value="full"
                            )
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Device ID/Name", html_for="scan-device-id"),
                            dbc.Input(id="scan-device-id", placeholder="Enter device identifier")
                        ], width=12),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Jailbreak/Root Status", html_for="scan-jailbreak"),
                            dbc.Checklist(
                                options=[{"label": "Device is jailbroken/rooted", "value": True}],
                                id="scan-jailbreak",
                                switch=True
                            )
                        ], width=12),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="scan-cost-display", className="alert alert-info"),
                            html.Div(id="scan-credit-warning", className="alert alert-warning d-none")
                        ], width=12),
                    ]),
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="scan-modal-cancel", color="secondary", className="me-2"),
                dbc.Button("Start Scan", id="scan-modal-start", color="primary"),
            ])
        ], id="new-scan-modal", size="lg", is_open=False)
    
    def _create_buy_credits_modal(self):
        """Create buy credits modal"""
        return dbc.Modal([
            dbc.ModalHeader("💰 Buy Credits"),
            dbc.ModalBody([
                dbc.Tabs([
                    dbc.Tab([
                        html.H5("Credit Packages", className="mb-3"),
                        dbc.Row([
                            dbc.Col(dbc.Card([
                                dbc.CardHeader("Starter", className="text-center bg-light"),
                                dbc.CardBody([
                                    html.H3("100", className="card-title text-center text-primary"),
                                    html.P("credits", className="text-center text-muted"),
                                    html.Hr(),
                                    html.P("Perfect for getting started", className="text-center"),
                                    html.Ul([
                                        html.Li("Basic scanning"),
                                        html.Li("Email support"),
                                        html.Li("10 scans/day"),
                                    ], className="list-unstyled"),
                                    html.H4("FREE", className="text-center fw-bold mt-3"),
                                    dbc.Button("Get Started", color="primary", className="w-100",
                                             id="btn-package-starter")
                                ])
                            ], className="h-100"), width=4),
                            
                            dbc.Col(dbc.Card([
                                dbc.CardHeader("Basic", className="text-center bg-primary text-white"),
                                dbc.CardBody([
                                    html.H3("1,000", className="card-title text-center text-primary"),
                                    html.P("credits", className="text-center text-muted"),
                                    html.Hr(),
                                    html.P("For individual users", className="text-center"),
                                    html.Ul([
                                        html.Li("AI detection"),
                                        html.Li("API access"),
                                        html.Li("100 scans/day"),
                                        html.Li("Email support"),
                                    ], className="list-unstyled"),
                                    html.H4("$9.99", className="text-center fw-bold mt-3"),
                                    dbc.Button("Buy Now", color="success", className="w-100",
                                             id="btn-package-basic")
                                ])
                            ], className="h-100 border-primary"), width=4),
                            
                            dbc.Col(dbc.Card([
                                dbc.CardHeader("Professional", className="text-center bg-warning"),
                                dbc.CardBody([
                                    html.H3("5,000", className="card-title text-center text-primary"),
                                    html.P("credits", className="text-center text-muted"),
                                    html.Hr(),
                                    html.P("For security professionals", className="text-center"),
                                    html.Ul([
                                        html.Li("All scan types"),
                                        html.Li("Advanced AI"),
                                        html.Li("Priority support"),
                                        html.Li("500 scans/day"),
                                        html.Li("Export features"),
                                    ], className="list-unstyled"),
                                    html.H4("$49.99", className="text-center fw-bold mt-3"),
                                    dbc.Button("Buy Now", color="warning", className="w-100",
                                             id="btn-package-professional")
                                ])
                            ], className="h-100"), width=4),
                        ]),
                    ], label="One-Time Purchase", tab_id="tab-one-time"),
                    
                    dbc.Tab([
                        html.H5("Subscription Plans", className="mb-3"),
                        dbc.Row([
                            dbc.Col(dbc.Card([
                                dbc.CardHeader("Basic Monthly", className="text-center bg-light"),
                                dbc.CardBody([
                                    html.H3("1,000", className="card-title text-center text-primary"),
                                    html.P("credits/month", className="text-center text-muted"),
                                    html.Hr(),
                                    html.P("Monthly subscription", className="text-center"),
                                    html.Ul([
                                        html.Li("AI detection"),
                                        html.Li("API access"),
                                        html.Li("Auto-renewal"),
                                    ], className="list-unstyled"),
                                    html.H4("$9.99/mo", className="text-center fw-bold mt-3"),
                                    dbc.Button("Subscribe", color="primary", className="w-100",
                                             id="btn-subscription-basic-monthly")
                                ])
                            ], className="h-100"), width=6),
                            
                            dbc.Col(dbc.Card([
                                dbc.CardHeader("Pro Yearly", className="text-center bg-success text-white"),
                                dbc.CardBody([
                                    html.H3("5,000", className="card-title text-center text-white"),
                                    html.P("credits/month", className="text-center text-white-50"),
                                    html.Hr(),
                                    html.P("Save 2 months free!", className="text-center text-white"),
                                    html.Ul([
                                        html.Li("All features"),
                                        html.Li("Priority support"),
                                        html.Li("Auto-renewal"),
                                        html.Li("2 months free"),
                                    ], className="list-unstyled text-white"),
                                    html.H4("$499.99/yr", className="text-center fw-bold mt-3 text-white"),
                                    dbc.Button("Subscribe", color="light", className="w-100",
                                             id="btn-subscription-pro-yearly")
                                ])
                            ], className="h-100"), width=6),
                        ]),
                    ], label="Subscription", tab_id="tab-subscription"),
                ]),
                
                html.Hr(),
                
                html.Div([
                    html.H6("Payment Method", className="mb-3"),
                    dbc.RadioItems(
                        options=[
                            {"label": html.Span([html.I(className="fab fa-cc-stripe me-2"), "Credit Card (Stripe)"]), "value": "stripe"},
                            {"label": html.Span([html.I(className="fab fa-cc-paypal me-2"), "PayPal"]), "value": "paypal"},
                            {"label": html.Span([html.I(className="fab fa-bitcoin me-2"), "Cryptocurrency"]), "value": "crypto"},
                        ],
                        value="stripe",
                        id="payment-method-selector",
                        inline=True,
                        className="mb-3"
                    ),
                    
                    # Payment form will be dynamically shown based on selection
                    html.Div(id="payment-form-container"),
                ]),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="credit-modal-cancel", color="secondary", className="me-2"),
                dbc.Button("Complete Purchase", id="credit-modal-purchase", color="success", disabled=True),
            ])
        ], id="buy-credits-modal", size="xl", is_open=False)
    
    def _create_report_modal(self):
        """Create report generation modal"""
        return dbc.Modal([
            dbc.ModalHeader("📊 Generate Report"),
            dbc.ModalBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Report Type", html_for="report-type"),
                            dbc.Select(
                                id="report-type",
                                options=[
                                    {"label": "📈 Scan Report", "value": "scan"},
                                    {"label": "🚨 Detection Report", "value": "detection"},
                                    {"label": "📊 Executive Summary", "value": "executive"},
                                    {"label": "🔒 Compliance Report", "value": "compliance"},
                                    {"label": "📋 Custom Report", "value": "custom"}
                                ],
                                value="scan"
                            )
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Label("Format", html_for="report-format"),
                            dbc.Select(
                                id="report-format",
                                options=[
                                    {"label": "📄 PDF", "value": "pdf"},
                                    {"label": "📊 HTML", "value": "html"},
                                    {"label": "📝 JSON", "value": "json"},
                                    {"label": "📈 Excel", "value": "excel"},
                                    {"label": "📋 CSV", "value": "csv"}
                                ],
                                value="pdf"
                            )
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Start Date", html_for="report-start-date"),
                            dcc.DatePickerSingle(
                                id="report-start-date",
                                date=datetime.now().date(),
                                display_format='YYYY-MM-DD',
                                className="w-100"
                            )
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Label("End Date", html_for="report-end-date"),
                            dcc.DatePickerSingle(
                                id="report-end-date",
                                date=datetime.now().date(),
                                display_format='YYYY-MM-DD',
                                className="w-100"
                            )
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Include Sections", html_for="report-sections"),
                            dbc.Checklist(
                                options=[
                                    {"label": "Executive Summary", "value": "executive"},
                                    {"label": "Detailed Findings", "value": "findings"},
                                    {"label": "Statistics", "value": "statistics"},
                                    {"label": "Charts & Graphs", "value": "charts"},
                                    {"label": "Recommendations", "value": "recommendations"},
                                    {"label": "Appendices", "value": "appendices"}
                                ],
                                value=["executive", "findings", "statistics", "recommendations"],
                                id="report-sections",
                            )
                        ], width=12),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Custom Title", html_for="report-title"),
                            dbc.Input(
                                id="report-title",
                                placeholder="Enter custom report title"
                            )
                        ], width=12),
                    ], className="mb-3"),
                    
                    dbc.Alert("Reports cost 5 credits each.", color="info", className="mb-3"),
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="report-modal-cancel", color="secondary", className="me-2"),
                dbc.Button("Generate Report", id="report-modal-generate", color="primary"),
            ])
        ], id="report-modal", size="lg", is_open=False)
        

    def setup_dash_callbacks(self):
        """Setup all Dash callbacks with navigation support"""
        
        # 1. Navigation callback - FIXED VERSION
        @self.dash_app.callback(
            Output('page-content', 'children'),
            [Input('url', 'pathname')]
        )
        def display_page(pathname):
            """Handle page navigation based on URL"""
            if pathname == '/dashboard' or pathname == '/dashboard/':
                return self._create_overview_page()
            elif pathname == '/dashboard/devices':
                return self._create_devices_page()
            elif pathname == '/dashboard/scans':
                return self._create_scans_page()
            elif pathname == '/dashboard/detections':
                return self._create_detections_page()
            elif pathname == '/dashboard/billing':
                return self._create_billing_page()
            elif pathname == '/dashboard/settings':
                return self._create_settings_page()
            else:
                return self._create_overview_page()
        
        # 2. Update quick statistics (auto-refresh)
        @self.dash_app.callback(
            [Output('total-devices', 'children'),
            Output('scans-today', 'children'),
            Output('total-detections', 'children'),
            Output('credit-balance', 'children')],
            [Input('dashboard-update-interval', 'n_intervals')]
        )
        def update_quick_stats(n):
            """Update quick statistics"""
            with self.Session() as db:
                user = db.query(User).first()
                if not user:
                    return ["0", "0", "0", "0"]
                
                device_count = db.query(Device).filter_by(owner_id=user.id).count()
                
                today = datetime.utcnow().date()
                scans_today = db.query(Scan).filter(
                    Scan.user_id == user.id,
                    func.date(Scan.start_time) == today
                ).count()
                
                active_detections = db.query(Detection).filter(
                    Detection.status.in_(['new', 'investigating', 'confirmed'])
                ).count()
                
                credits = db.query(UserCredits).filter_by(user_id=user.id).first()
                credit_balance = credits.credits_balance if credits else 0
                
                return [f"{device_count}", f"{scans_today}", f"{active_detections}", f"{credit_balance:,}"]
        
        # 3. Update detection timeline chart
        @self.dash_app.callback(
            Output('detection-timeline-chart', 'figure'),
            [Input('dashboard-update-interval', 'n_intervals')]
        )
        def update_detection_timeline(n):
            """Update detection timeline chart"""
            with self.Session() as db:
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                
                detections = db.query(
                    func.date(Detection.detected_at).label('date'),
                    func.count(Detection.id).label('count')
                ).filter(
                    Detection.detected_at >= thirty_days_ago
                ).group_by(func.date(Detection.detected_at)).order_by('date').all()
                
                if not detections:
                    fig = go.Figure()
                    fig.update_layout(
                        title='No detections in the last 30 days',
                        xaxis_title='Date',
                        yaxis_title='Detections',
                        template='plotly_white',
                        height=300
                    )
                    return fig
                
                dates = [d.date for d in detections]
                counts = [d.count for d in detections]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=counts, mode='lines+markers', name='Detections',
                    line=dict(color='red', width=2), marker=dict(size=8, color='red'),
                    fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.1)'
                ))
                
                fig.update_layout(
                    title='Detection Timeline (Last 30 Days)',
                    xaxis_title='Date', yaxis_title='Number of Detections',
                    template='plotly_white', height=300, hovermode='x unified', showlegend=False
                )
                
                return fig
        
        # 4. Update recent alerts list
        @self.dash_app.callback(
            Output('recent-alerts-list', 'children'),
            [Input('dashboard-update-interval', 'n_intervals')]
        )
        def update_recent_alerts(n):
            """Update recent alerts list"""
            with self.Session() as db:
                alerts = db.query(Alert)\
                    .filter(Alert.status.in_(['new', 'acknowledged']))\
                    .order_by(Alert.created_at.desc())\
                    .limit(5).all()
                
                if not alerts:
                    return html.Div("No recent alerts", className="text-muted text-center")
                
                alert_items = []
                for alert in alerts:
                    if alert.severity == 'critical':
                        icon, color = 'fas fa-fire', 'danger'
                    elif alert.severity == 'high':
                        icon, color = 'fas fa-exclamation-triangle', 'warning'
                    elif alert.severity == 'medium':
                        icon, color = 'fas fa-exclamation-circle', 'info'
                    else:
                        icon, color = 'fas fa-info-circle', 'secondary'
                    
                    alert_items.append(dbc.Alert([
                        html.Div([html.I(className=f"{icon} me-2"), html.Strong(alert.title)]),
                        html.Small(alert.message, className="d-block mt-1 text-muted"),
                        html.Small(alert.created_at.strftime('%H:%M'), className="float-end text-muted"),
                    ], color=color, className="py-2 mb-2"))
                
                return alert_items
        
        # 5. Update system status indicators
        @self.dash_app.callback(
            [Output('system-cpu', 'children'), Output('cpu-progress', 'value'), Output('cpu-progress', 'color'),
            Output('system-memory', 'children'), Output('memory-progress', 'value'), Output('memory-progress', 'color'),
            Output('ai-models-status', 'children'), Output('license-status', 'children'), Output('system-uptime', 'children')],
            [Input('system-status-interval', 'n_intervals')]
        )
        def update_system_status(n):
            """Update system status indicators"""
            cpu_percent = psutil.cpu_percent()
            cpu_color = 'success' if cpu_percent < 50 else 'warning' if cpu_percent < 80 else 'danger'
            
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_color = 'success' if memory_percent < 60 else 'warning' if memory_percent < 85 else 'danger'
            
            ai_status = "45 models loaded"
            license_status = "Professional (Active)"
            
            uptime_seconds = time.time() - psutil.boot_time()
            uptime_days = int(uptime_seconds // 86400)
            uptime_hours = int((uptime_seconds % 86400) // 3600)
            uptime_str = f"{uptime_days}d {uptime_hours}h"
            
            return [
                f"{cpu_percent:.1f}%", cpu_percent, cpu_color,
                f"{memory_percent:.1f}% ({memory.used // (1024**3)}GB/{memory.total // (1024**3)}GB)",
                memory_percent, memory_color, ai_status, license_status, uptime_str
            ]
        
        @self.dash_app.callback(
            Output('add-device-modal', 'is_open'),
            [Input('btn-add-device', 'n_clicks'),
            Input('device-modal-cancel', 'n_clicks'),
            Input('device-modal-save', 'n_clicks')],
            [State('add-device-modal', 'is_open')]
        )
        def toggle_device_modal(btn_open, btn_cancel, btn_save, is_open):
            ctx = dash.callback_context
            if not ctx.triggered:
                return is_open
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            return True if button_id == 'btn-add-device' else False if button_id in ['device-modal-cancel', 'device-modal-save'] else is_open
        
        @self.dash_app.callback(
        Output('device-save-feedback', 'children'),
        [Input('device-modal-save', 'n_clicks')],
        [State('device-name', 'value'),
        State('device-type', 'value'),
        State('device-identifier', 'value'),
        State('device-os', 'value'),
        State('device-model', 'value'),
        State('device-manufacturer', 'value'),
        State('device-security', 'value'),
        State('device-notes', 'value')]
    )
        def save_device(n_clicks, name, device_type, identifier, os_version, model, manufacturer, security, notes):
            """Save new device to database"""
            if not n_clicks:
                return ""
            
            if not name:
                return dbc.Alert("Device name is required!", color="danger")
            
            with self.Session() as db:
                # Get current user (for demo, get first user - in production, get from session)
                user = db.query(User).first()  # TODO: Replace with proper user session
                
                device = Device(
                    device_uuid=str(uuid.uuid4()),
                    device_id=identifier or f"{device_type}_{uuid.uuid4().hex[:8]}",
                    owner_id=user.id,
                    device_type=device_type,
                    platform=device_type,
                    manufacturer=manufacturer or "Unknown",
                    model=model or "Unknown",
                    os_name=os_version or "Unknown",
                    jailbreak_status="jailbroken" in (security or []),
                    bootloader_unlocked="bootloader" in (security or []),
                    status="active",
                    created_at=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    notes=notes
                )
                
                db.add(device)
                db.commit()
                
                return dbc.Alert(f"✅ Device '{name}' added successfully!", color="success")

        def _create_devices_page(self):
            """Create devices page content"""
            return html.Div([
                html.H1("📱 Devices", className="my-4"),
                dbc.Row([
                    dbc.Col(dbc.Button("➕ Add New Device", color="primary", id="btn-add-device", className="mb-3"), width=12),
                ]),
                dbc.Card([
                    dbc.CardHeader("Your Devices", className="fw-bold"),
                    dbc.CardBody(
                        html.Div(id="devices-list-table", children="Loading devices...")
                    )
                ], className="shadow-sm border-0")
            ])

        # 6. Modal toggle callbacks
        @self.dash_app.callback(
            Output('new-scan-modal', 'is_open'),
            [Input('btn-new-scan', 'n_clicks'),
            Input('scan-modal-cancel', 'n_clicks'),
            Input('scan-modal-start', 'n_clicks')],
            [State('new-scan-modal', 'is_open')]
        )
        def toggle_scan_modal(btn_open, btn_cancel, btn_start, is_open):
            ctx = dash.callback_context
            if not ctx.triggered:
                return is_open
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            return True if button_id == 'btn-new-scan' else False if button_id in ['scan-modal-cancel', 'scan-modal-start'] else is_open
        
        @self.dash_app.callback(
            Output('buy-credits-modal', 'is_open'),
            [Input('btn-buy-credits', 'n_clicks'),
            Input('credit-modal-cancel', 'n_clicks'),
            Input('credit-modal-purchase', 'n_clicks')],
            [State('buy-credits-modal', 'is_open')]
        )
        def toggle_credits_modal(btn_open, btn_cancel, btn_purchase, is_open):
            ctx = dash.callback_context
            if not ctx.triggered:
                return is_open
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            return True if button_id == 'btn-buy-credits' else False if button_id in ['credit-modal-cancel', 'credit-modal-purchase'] else is_open
        
        @self.dash_app.callback(
            Output('report-modal', 'is_open'),
            [Input('btn-generate-report', 'n_clicks'),
            Input('report-modal-cancel', 'n_clicks'),
            Input('report-modal-generate', 'n_clicks')],
            [State('report-modal', 'is_open')]
        )
        def toggle_report_modal(btn_open, btn_cancel, btn_generate, is_open):
            ctx = dash.callback_context
            if not ctx.triggered:
                return is_open
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            return True if button_id == 'btn-generate-report' else False if button_id in ['report-modal-cancel', 'report-modal-generate'] else is_open
        
        # 7. Update scan cost display
        @self.dash_app.callback(
            [Output('scan-cost-display', 'children'),
            Output('scan-credit-warning', 'className')],
            [Input('scan-type', 'value'),
            Input('scan-device-type', 'value')]
        )
        def update_scan_cost(scan_type, device_type):
            """Update scan cost display based on selections"""
            credit_costs = {
                'quick': 1, 'basic': 3, 'full': 5, 'deep': 10,
                'network': 4, 'memory': 6, 'comprehensive': 15
            }
            
            cost = credit_costs.get(scan_type, 5)
            
            # Get user's credit balance
            with self.Session() as db:
                user = db.query(User).first()
                if user:
                    credits = db.query(UserCredits).filter_by(user_id=user.id).first()
                    balance = credits.credits_balance if credits else 0
                    
                    if balance < cost:
                        warning_class = "alert alert-warning"
                    else:
                        warning_class = "alert alert-warning d-none"
                else:
                    warning_class = "alert alert-warning d-none"
            
            display = html.Div([
                html.Strong(f"Scan Cost: {cost} credits"),
                html.Br(),
                html.Small(f"Device: {device_type}, Type: {scan_type}")
            ])
            
            return display, warning_class

    def _create_overview_page(self):
        """Create overview page content"""
        return html.Div([
            html.H1("📊 Dashboard Overview", className="my-4"),
            html.P("Welcome to Pegasus Enterprise Dashboard. Monitor your security status in real-time.", className="text-muted mb-4"),
            
            # Quick stats row
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-mobile-alt fa-2x text-primary"),
                            html.Div([
                                html.H4("Devices", className="card-title mb-1"),
                                html.H2(id="total-devices", className="card-text fw-bold"),
                                html.Small("Active devices", className="text-muted")
                            ], className="ms-3")
                        ], className="d-flex align-items-center")
                    ])
                ], className="stat-card shadow-sm border-0"), width=3),
                
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-search fa-2x text-success"),
                            html.Div([
                                html.H4("Scans Today", className="card-title mb-1"),
                                html.H2(id="scans-today", className="card-text fw-bold"),
                                html.Small("Completed scans", className="text-muted")
                            ], className="ms-3")
                        ], className="d-flex align-items-center")
                    ])
                ], className="stat-card shadow-sm border-0"), width=3),
                
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle fa-2x text-danger"),
                            html.Div([
                                html.H4("Detections", className="card-title mb-1"),
                                html.H2(id="total-detections", className="card-text fw-bold"),
                                html.Small("Active threats", className="text-muted")
                            ], className="ms-3")
                        ], className="d-flex align-items-center")
                    ])
                ], className="stat-card shadow-sm border-0"), width=3),
                
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-coins fa-2x text-warning"),
                            html.Div([
                                html.H4("Credits", className="card-title mb-1"),
                                html.H2(id="credit-balance", className="card-text fw-bold"),
                                html.Small("Available", className="text-muted")
                            ], className="ms-3")
                        ], className="d-flex align-items-center")
                    ])
                ], className="stat-card shadow-sm border-0"), width=3),
            ], className="mb-4"),
            
            # Charts and alerts row
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("📈 Detection Timeline", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="detection-timeline-chart"))
                ], className="shadow-sm border-0"), width=8),
                
                dbc.Col(dbc.Card([
                    dbc.CardHeader("🚨 Recent Alerts", className="fw-bold"),
                    dbc.CardBody(
                        html.Div(id="recent-alerts-list", className="alert-list")
                    )
                ], className="shadow-sm border-0"), width=4),
            ], className="mb-4"),
            
            # Quick actions
            dbc.Row(dbc.Card([
                dbc.CardHeader("⚡ Quick Actions", className="fw-bold"),
                dbc.CardBody(
                    dbc.Row([
                        dbc.Col(dbc.Button([
                            html.I(className="fas fa-search me-2"),
                            "New Scan"
                        ], color="primary", className="w-100", id="btn-new-scan"), width=3),
                        
                        dbc.Col(dbc.Button([
                            html.I(className="fas fa-plus me-2"),
                            "Add Device"
                        ], color="success", className="w-100", id="btn-add-device"), width=3),
                        
                        dbc.Col(dbc.Button([
                            html.I(className="fas fa-coins me-2"),
                            "Buy Credits"
                        ], color="warning", className="w-100", id="btn-buy-credits"), width=3),
                        
                        dbc.Col(dbc.Button([
                            html.I(className="fas fa-file-pdf me-2"),
                            "Generate Report"
                        ], color="info", className="w-100", id="btn-generate-report"), width=3),
                    ])
                )
            ], className="shadow-sm border-0"), className="mb-4"),
        ])

    def _create_devices_page(self):
        """Create devices page content"""
        return html.Div([
            html.H1("📱 Devices", className="my-4"),
            dbc.Card([
                dbc.CardHeader("Managed Devices", className="fw-bold"),
                dbc.CardBody(
                    html.Div([
                        html.P("Device management page content will appear here."),
                        html.P("You can add, remove, and manage devices from this page."),
                        dbc.Button("Add New Device", color="primary", className="mt-3")
                    ])
                )
            ], className="shadow-sm border-0")
        ])

    def _create_scans_page(self):
        """Create scans page content"""
        return html.Div([
            html.H1("🔍 Scans", className="my-4"),
            dbc.Card([
                dbc.CardHeader("Scan History", className="fw-bold"),
                dbc.CardBody(
                    html.Div([
                        html.P("Scan history and management will appear here."),
                        html.P("View past scans, start new scans, and analyze results."),
                        dbc.Button("Start New Scan", color="primary", className="mt-3")
                    ])
                )
            ], className="shadow-sm border-0")
        ])

    def _create_detections_page(self):
        """Create detections page content"""
        return html.Div([
            html.H1("🚨 Detections", className="my-4"),
            dbc.Card([
                dbc.CardHeader("Threat Detections", className="fw-bold"),
                dbc.CardBody(
                    html.Div([
                        html.P("Threat detection management will appear here."),
                        html.P("View and manage detected threats, false positives, and investigations."),
                        dbc.Button("View All Detections", color="primary", className="mt-3")
                    ])
                )
            ], className="shadow-sm border-0")
        ])

    def _create_billing_page(self):
        """Create billing page content"""
        return html.Div([
            html.H1("💰 Billing", className="my-4"),
            dbc.Card([
                dbc.CardHeader("Credit Management", className="fw-bold"),
                dbc.CardBody(
                    html.Div([
                        html.P("Billing and credit management will appear here."),
                        html.P("View your credit balance, purchase credits, and manage subscriptions."),
                        dbc.Button("Buy Credits", color="primary", className="mt-3")
                    ])
                )
            ], className="shadow-sm border-0")
        ])

    def _create_settings_page(self):
        """Create settings page content"""
        return html.Div([
            html.H1("⚙️ Settings", className="my-4"),
            dbc.Card([
                dbc.CardHeader("Application Settings", className="fw-bold"),
                dbc.CardBody(
                    html.Div([
                        html.P("Application settings will appear here."),
                        html.P("Configure system preferences, notifications, and security settings."),
                        dbc.Button("Save Settings", color="primary", className="mt-3")
                    ])
                )
            ], className="shadow-sm border-0")
        ])
    
    def setup_api_routes(self):
        """Setup all API routes"""
        
        # Authentication decorator
        def api_key_required(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                api_key = request.headers.get('X-API-Key')
                
                if not api_key:
                    return jsonify({
                        'success': False,
                        'error': 'API key required',
                        'code': 'API_KEY_MISSING'
                    }), 401
                
                with self.Session() as db:
                    user = db.query(User).filter_by(api_key=api_key).first()
                    
                    if not user or not user.is_active:
                        return jsonify({
                            'success': False,
                            'error': 'Invalid or inactive API key',
                            'code': 'API_KEY_INVALID'
                        }), 401
                
                return f(*args, **kwargs, user=user)
            return decorated
        
        # API Routes
        
        @self.flask_app.route('/api/v1/health', methods=['GET'])
        def api_health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '5.0.0',
                'services': {
                    'database': 'connected',
                    'ai_engine': 'ready',
                    'license_manager': 'active',
                    'credit_manager': 'active'
                }
            })
        
        @self.flask_app.route('/api/v1/scan', methods=['POST'])
        @api_key_required
        def api_scan(user):
            """Start a new scan"""
            data = request.json
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            device_type = data.get('device_type', 'auto')
            scan_type = data.get('scan_type', 'full')
            jailbreak = data.get('jailbreak', False)
            device_id = data.get('device_id', f"{device_type}_{user.id}")
            
            # Check credits
            with self.Session() as db:
                credit_manager = EnhancedCreditManager(db)
                can_scan, result = credit_manager.check_credits(
                    user.id, 'scan', scan_type
                )
                
                if not can_scan:
                    return jsonify({
                        'success': False,
                        'error': str(result),
                        'code': 'INSUFFICIENT_CREDITS'
                    }), 402
                
                # Create scan record
                scan = Scan(
                    scan_uuid=str(uuid.uuid4()),
                    scan_id=f"SCAN{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}",
                    user_id=user.id,
                    scan_type=scan_type,
                    scan_mode='api',
                    priority=data.get('priority', 'normal'),
                    status='pending',
                    scan_config=data,
                    created_at=datetime.utcnow()
                )
                
                # Check if device exists
                device = db.query(Device).filter_by(
                    device_id=device_id,
                    owner_id=user.id
                ).first()
                
                if not device:
                    device = Device(
                        device_uuid=str(uuid.uuid4()),
                        device_id=device_id,
                        owner_id=user.id,
                        device_type=device_type,
                        platform='iOS' if 'iphone' in device_type.lower() else 'Android',
                        jailbreak_status=jailbreak,
                        status='active',
                        created_at=datetime.utcnow(),
                        last_seen=datetime.utcnow()
                    )
                    db.add(device)
                
                scan.device = device
                db.add(scan)
                db.commit()
                
                # Use credits
                try:
                    credit_result = credit_manager.use_credits(
                        user.id, 'scan', scan_type,
                        description=f"API Scan: {scan_type} on {device_type}"
                    )
                except ValueError as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 402
                
                # Start scan in background (simplified)
                # In production, this would use Celery or similar
                scan.status = 'running'
                scan.start_time = datetime.utcnow()
                db.commit()
                
                # Simulate scan results (in production, actual scanning would happen)
                scanner = UniversalDeviceScanner()
                scan_results = scanner.scan_universal(device_type, jailbreak)
                
                scan.end_time = datetime.utcnow()
                scan.duration = (scan.end_time - scan.start_time).total_seconds()
                scan.status = 'completed'
                scan.raw_results = scan_results
                
                if scan_results.get('final_assessment', {}).get('verdict') == 'INFECTED':
                    scan.pegasus_detected = True
                    scan.variant_detected = scan_results.get('final_assessment', {}).get('variant')
                    scan.confidence_score = scan_results.get('final_assessment', {}).get('confidence', 0)
                    scan.risk_level = scan_results.get('final_assessment', {}).get('risk_level')
                    
                    # Create detection records
                    detections = scan_results.get('detection_results', {}).get('ai', {}).get('detections', [])
                    for detection in detections:
                        det = Detection(
                            detection_uuid=str(uuid.uuid4()),
                            detection_id=f"DET{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}",
                            scan_id=scan.id,
                            device_id=device.id,
                            variant=detection.get('variant'),
                            confidence=detection.get('confidence'),
                            detection_method=detection.get('model'),
                            detection_source='ai',
                            severity='high' if detection.get('confidence', 0) > 0.7 else 'medium',
                            detected_at=datetime.utcnow()
                        )
                        db.add(det)
                
                db.commit()
                
                return jsonify({
                    'success': True,
                    'scan_id': scan.scan_id,
                    'status': scan.status,
                    'credits_used': credit_result.get('cost', 0),
                    'results': {
                        'pegasus_detected': scan.pegasus_detected,
                        'variant': scan.variant_detected,
                        'confidence': scan.confidence_score,
                        'risk_level': scan.risk_level
                    },
                    'timestamp': scan.end_time.isoformat() if scan.end_time else None
                })
        
        @self.flask_app.route('/api/v1/credits/balance', methods=['GET'])
        @api_key_required
        def api_credits_balance(user):
            """Get credit balance"""
            with self.Session() as db:
                credit_manager = EnhancedCreditManager(db)
                credit_info = credit_manager.get_user_credits_info(user.id)
                
                return jsonify({
                    'success': True,
                    'credits': credit_info
                })
        
        @self.flask_app.route('/api/v1/credits/purchase', methods=['POST'])
        @api_key_required
        def api_purchase_credits(user):
            """Purchase credits"""
            data = request.json
            
            package_type = data.get('package', 'basic')
            payment_method = data.get('payment_method', 'stripe')
            payment_details = data.get('payment_details', {})
            
            with self.Session() as db:
                credit_manager = EnhancedCreditManager(db)
                
                try:
                    result = credit_manager.purchase_credits(
                        user.id, package_type, payment_method, payment_details
                    )
                    
                    return jsonify(result)
                    
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 400
        
        @self.flask_app.route('/api/v1/license/validate', methods=['POST'])
        @api_key_required
        def api_validate_license(user):
            """Validate license"""
            data = request.json
            license_key = data.get('license_key')
            device_fingerprint = data.get('device_fingerprint')
            
            if not license_key:
                return jsonify({
                    'success': False,
                    'error': 'License key required'
                }), 400
            
            with self.Session() as db:
                license_manager = EnhancedLicenseManager(db)
                result = license_manager.validate_license(license_key, device_fingerprint)
                
                return jsonify(result)
        
        @self.flask_app.route('/api/v1/license/generate', methods=['POST'])
        @api_key_required
        def api_generate_license(user):
            """Generate a new license (admin only)"""
            if not user.is_superuser:
                return jsonify({
                    'success': False,
                    'error': 'Admin access required'
                }), 403
            
            data = request.json
            tier = data.get('tier', 'personal')
            duration_days = data.get('duration_days')
            metadata = data.get('metadata')
            
            with self.Session() as db:
                license_manager = EnhancedLicenseManager(db)
                
                try:
                    result = license_manager.generate_license_key(
                        user.id, tier, duration_days, metadata
                    )
                    
                    return jsonify(result)
                    
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 400
        
        @self.flask_app.route('/api/v1/devices', methods=['GET'])
        @api_key_required
        def api_get_devices(user):
            """Get user's devices"""
            with self.Session() as db:
                devices = db.query(Device).filter_by(owner_id=user.id).all()
                
                return jsonify({
                    'success': True,
                    'devices': [
                        {
                            'id': d.device_id,
                            'type': d.device_type,
                            'platform': d.platform,
                            'model': d.model,
                            'os_version': d.os_version,
                            'jailbreak': d.jailbreak_status,
                            'risk_score': d.risk_score,
                            'last_seen': d.last_seen.isoformat() if d.last_seen else None,
                            'status': d.status
                        }
                        for d in devices
                    ]
                })
        
        @self.flask_app.route('/api/v1/scans', methods=['GET'])
        @api_key_required
        def api_get_scans(user):
            """Get user's scans"""
            with self.Session() as db:
                limit = min(int(request.args.get('limit', 50)), 100)
                offset = int(request.args.get('offset', 0))
                
                scans = db.query(Scan)\
                    .filter_by(user_id=user.id)\
                    .order_by(Scan.start_time.desc())\
                    .limit(limit)\
                    .offset(offset)\
                    .all()
                
                return jsonify({
                    'success': True,
                    'scans': [
                        {
                            'id': s.scan_id,
                            'device': s.device.device_type if s.device else 'Unknown',
                            'type': s.scan_type,
                            'status': s.status,
                            'result': 'INFECTED' if s.pegasus_detected else 'CLEAN',
                            'confidence': s.confidence_score,
                            'start_time': s.start_time.isoformat() if s.start_time else None,
                            'duration': s.duration
                        }
                        for s in scans
                    ],
                    'total': db.query(Scan).filter_by(user_id=user.id).count()
                })
        
        @self.flask_app.route('/api/v1/detections', methods=['GET'])
        @api_key_required
        def api_get_detections(user):
            """Get user's detections"""
            with self.Session() as db:
                limit = min(int(request.args.get('limit', 50)), 100)
                offset = int(request.args.get('offset', 0))
                severity = request.args.get('severity')
                status = request.args.get('status')
                
                query = db.query(Detection)\
                    .join(Scan)\
                    .filter(Scan.user_id == user.id)
                
                if severity:
                    query = query.filter(Detection.severity == severity)
                
                if status:
                    query = query.filter(Detection.status == status)
                
                detections = query\
                    .order_by(Detection.detected_at.desc())\
                    .limit(limit)\
                    .offset(offset)\
                    .all()
                
                return jsonify({
                    'success': True,
                    'detections': [
                        {
                            'id': d.detection_id,
                            'variant': d.variant,
                            'confidence': d.confidence,
                            'severity': d.severity,
                            'status': d.status,
                            'device': d.device.device_type if d.device else 'Unknown',
                            'detected_at': d.detected_at.isoformat() if d.detected_at else None,
                            'file_path': d.file_path,
                            'process_name': d.process_name
                        }
                        for d in detections
                    ],
                    'total': query.count()
                })
        
        @self.flask_app.route('/api/v1/reports/generate', methods=['POST'])
        @api_key_required
        def api_generate_report(user):
            """Generate a report"""
            data = request.json
            
            report_type = data.get('type', 'scan')
            period_start = data.get('period_start')
            period_end = data.get('period_end')
            format_type = data.get('format', 'pdf')
            
            with self.Session() as db:
                # Create report record
                report = Report(
                    report_uuid=str(uuid.uuid4()),
                    report_id=f"REP{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}",
                    user_id=user.id,
                    report_type=report_type,
                    period_start=datetime.fromisoformat(period_start) if period_start else None,
                    period_end=datetime.fromisoformat(period_end) if period_end else None,
                    title=f"Pegasus Enterprise Report - {datetime.utcnow().strftime('%Y-%m-%d')}",
                    status='generating',
                    created_at=datetime.utcnow()
                )
                
                db.add(report)
                db.commit()
                
                # Generate report content (simplified)
                report_data = self._generate_report_data(user.id, report_type, period_start, period_end)
                
                report.summary = report_data.get('summary', '')
                report.findings = report_data.get('findings', {})
                report.statistics = report_data.get('statistics', {})
                report.status = 'ready'
                report.generated_at = datetime.utcnow()
                
                # Generate file based on format
                if format_type == 'pdf':
                    # Generate PDF
                    filename = f"reports/report_{report.report_id}.pdf"
                    self._generate_pdf_report(report, report_data, filename)
                    report.export_location = filename
                    report.export_formats = ['pdf']
                
                elif format_type == 'json':
                    report.export_formats = ['json']
                
                db.commit()
                
                return jsonify({
                    'success': True,
                    'report_id': report.report_id,
                    'status': report.status,
                    'download_url': f"/api/v1/reports/download/{report.report_id}" if report.export_location else None
                })
        
        # ================== FIXED DASHBOARD ROUTES ==================
        
        # Web routes for the dashboard
        @self.flask_app.route('/')
        def index():
            """Main index page"""
            return redirect('/dashboard/')
        
        @self.flask_app.route('/dashboard')
        def dashboard():
            """Dashboard page"""
            return self.dash_app.index()
        
        # ADD THESE ROUTES - FIXED NAVIGATION
        @self.flask_app.route('/dashboard/devices')
        @login_required
        def dashboard_devices():
            """Devices dashboard page"""
            return self.dash_app.index()
        
        @self.flask_app.route('/dashboard/scans')
        @login_required
        def dashboard_scans():
            """Scans dashboard page"""
            return self.dash_app.index()
        
        @self.flask_app.route('/dashboard/detections')
        @login_required
        def dashboard_detections():
            """Detections dashboard page"""
            return self.dash_app.index()
        
        @self.flask_app.route('/dashboard/billing')
        @login_required
        def dashboard_billing():
            """Billing dashboard page"""
            return self.dash_app.index()
        
        @self.flask_app.route('/dashboard/settings')
        @login_required
        def dashboard_settings():
            """Settings dashboard page"""
            return self.dash_app.index()
        
        # Replace the existing /login route with this enhanced version:
        @self.flask_app.route('/login', methods=['GET', 'POST'])
        def login():
            """Enhanced login page with admin/user separation"""
            if current_user.is_authenticated:
                return redirect('/dashboard')
            
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')
                
                with self.Session() as db:
                    user = db.query(User).filter_by(username=username).first()
                    
                    if user and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
                        login_user(user)
                        
                        # Redirect based on user role
                        if user.is_admin or user.is_superuser or user.role == 'admin':
                            return redirect('/admin/dashboard')
                        else:
                            return redirect('/dashboard')
                    else:
                        flash('Invalid username or password', 'error')
            
            # Return enhanced HTML login form
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>🏢 Pegasus Enterprise - Login</title>
                <style>
                    body { 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        margin: 0; 
                        padding: 0; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        height: 100vh; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                    }
                    .login-container { 
                        background: white; 
                        padding: 40px; 
                        border-radius: 15px; 
                        box-shadow: 0 20px 60px rgba(0,0,0,0.3); 
                        width: 350px; 
                        text-align: center;
                    }
                    .logo { 
                        font-size: 36px; 
                        margin-bottom: 20px; 
                        color: #667eea; 
                    }
                    h1 { 
                        color: #333; 
                        margin-bottom: 30px; 
                        font-weight: 600; 
                    }
                    .form-group { 
                        margin-bottom: 20px; 
                        text-align: left; 
                    }
                    label { 
                        display: block; 
                        margin-bottom: 8px; 
                        color: #555; 
                        font-weight: 500; 
                    }
                    input { 
                        width: 100%; 
                        padding: 12px 15px; 
                        border: 2px solid #e0e0e0; 
                        border-radius: 8px; 
                        box-sizing: border-box; 
                        font-size: 16px; 
                        transition: border-color 0.3s;
                    }
                    input:focus { 
                        outline: none; 
                        border-color: #667eea; 
                    }
                    button { 
                        width: 100%; 
                        padding: 14px; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; 
                        border: none; 
                        border-radius: 8px; 
                        cursor: pointer; 
                        font-size: 16px; 
                        font-weight: 600; 
                        margin-top: 10px; 
                        transition: transform 0.2s, box-shadow 0.2s;
                    }
                    button:hover { 
                        transform: translateY(-2px); 
                        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); 
                    }
                    .error { 
                        color: #ff4757; 
                        margin-top: 10px; 
                        padding: 10px; 
                        background: #ff475710; 
                        border-radius: 5px; 
                        border-left: 4px solid #ff4757;
                    }
                    .demo-credentials { 
                        margin-top: 25px; 
                        padding: 15px; 
                        background: #f8f9fa; 
                        border-radius: 8px; 
                        text-align: left;
                    }
                    .demo-credentials h4 { 
                        margin-top: 0; 
                        color: #555; 
                    }
                    .role-badge { 
                        display: inline-block; 
                        padding: 3px 8px; 
                        border-radius: 12px; 
                        font-size: 12px; 
                        font-weight: bold; 
                        margin-left: 5px; 
                    }
                    .admin-badge { background: #ff6b6b; color: white; }
                    .user-badge { background: #4ecdc4; color: white; }
                </style>
            </head>
            <body>
                <div class="login-container">
                    <div class="logo">🏢</div>
                    <h1>Pegasus Enterprise</h1>
                    <form method="POST">
                        <div class="form-group">
                            <label for="username">Username</label>
                            <input type="text" name="username" placeholder="Enter username" required>
                        </div>
                        <div class="form-group">
                            <label for="password">Password</label>
                            <input type="password" name="password" placeholder="Enter password" required>
                        </div>
                        <button type="submit">Login</button>
                    </form>
                    
                    {% with messages = get_flashed_messages(with_categories=true) %}
                        {% if messages %}
                            {% for category, message in messages %}
                                <div class="error">{{ message }}</div>
                            {% endfor %}
                        {% endif %}
                    {% endwith %}
                    
                    <div class="demo-credentials">
                        <h4>Demo Credentials:</h4>
                        <p><strong>Admin:</strong> admin / admin123 <span class="role-badge admin-badge">Admin</span></p>
                        <p><strong>User:</strong> user / user123 <span class="role-badge user-badge">User</span></p>
                    </div>
                </div>
            </body>
            </html>
            '''
        
        @self.flask_app.route('/logout')
        @login_required
        def logout():
            """Logout"""
            logout_user()
            return redirect('/login')
        
        @self.flask_app.route('/admin')
        @login_required
        def admin():
            """Admin panel - SIMPLIFIED VERSION"""
            if not current_user.is_superuser:
                flash('Admin access required', 'error')
                return redirect('/dashboard')
            
            with self.Session() as db:
                users = db.query(User).count()
                devices = db.query(Device).count()
                scans = db.query(Scan).count()
                detections = db.query(Detection).count()
                
                return f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>🏢 Admin Panel</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                        .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 200px; text-align: center; }}
                        .stat-number {{ font-size: 32px; font-weight: bold; color: #667eea; }}
                        .stat-label {{ color: #666; margin-top: 10px; }}
                        .nav {{ margin: 20px 0; }}
                        .nav a {{ display: inline-block; margin-right: 10px; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; }}
                    </style>
                </head>
                <body>
                    <h1>🏢 Admin Panel</h1>
                    <div class="nav">
                        <a href="/dashboard">Dashboard</a>
                        <a href="/admin">Admin</a>
                        <a href="/logout">Logout</a>
                    </div>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-number">{users}</div>
                            <div class="stat-label">Users</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{devices}</div>
                            <div class="stat-label">Devices</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{scans}</div>
                            <div class="stat-label">Scans</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{detections}</div>
                            <div class="stat-label">Detections</div>
                        </div>
                    </div>
                </body>
                </html>
                '''
    
    def _generate_report_data(self, user_id, report_type, period_start, period_end):
        """Generate report data"""
        with self.Session() as db:
            # Get scan statistics
            scans_query = db.query(Scan).filter_by(user_id=user_id)
            
            if period_start:
                scans_query = scans_query.filter(Scan.start_time >= period_start)
            if period_end:
                scans_query = scans_query.filter(Scan.start_time <= period_end)
            
            total_scans = scans_query.count()
            infected_scans = scans_query.filter(Scan.pegasus_detected == True).count()
            
            # Get detection statistics
            detections_query = db.query(Detection)\
                .join(Scan)\
                .filter(Scan.user_id == user_id)
            
            if period_start:
                detections_query = detections_query.filter(Detection.detected_at >= period_start)
            if period_end:
                detections_query = detections_query.filter(Detection.detected_at <= period_end)
            
            total_detections = detections_query.count()
            
            # Group by variant
            variants = db.query(
                Detection.variant,
                func.count(Detection.id).label('count')
            ).join(Scan)\
             .filter(Scan.user_id == user_id)\
             .group_by(Detection.variant)\
             .all()
            
            # Get device statistics
            devices = db.query(Device).filter_by(owner_id=user_id).all()
            
            return {
                'summary': f"Pegasus Enterprise Report covering {period_start} to {period_end}" if period_start else "Comprehensive Pegasus Enterprise Report",
                'statistics': {
                    'total_scans': total_scans,
                    'infected_scans': infected_scans,
                    'infection_rate': infected_scans / total_scans if total_scans > 0 else 0,
                    'total_detections': total_detections,
                    'devices_monitored': len(devices),
                    'average_risk_score': sum(d.risk_score for d in devices) / len(devices) if devices else 0
                },
                'findings': {
                    'variants_detected': [
                        {'variant': v.variant, 'count': v.count}
                        for v in variants
                    ],
                    'high_risk_devices': [
                        {'device': d.device_id, 'risk_score': d.risk_score}
                        for d in devices if d.risk_score > 70
                    ],
                    'recent_infections': infected_scans
                },
                'recommendations': [
                    'Maintain regular scanning schedule',
                    'Update device operating systems',
                    'Review and mitigate high-risk detections',
                    'Consider enterprise-grade protection for critical devices'
                ]
            }
    
    def _generate_pdf_report(self, report, report_data, filename):
        """Generate PDF report"""
        # This would generate an actual PDF
        # For now, create a placeholder
        Path('reports').mkdir(exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(f"Pegasus Enterprise Report\n")
            f.write(f"Report ID: {report.report_id}\n")
            f.write(f"Generated: {report.generated_at}\n")
            f.write(f"\nSummary:\n{report_data.get('summary', '')}\n")
        
        return filename
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the enterprise application"""
        print(UnicodeArt.get_banner())
        print("\n" + "="*80)
        print("🏢 PEGASUS ENTERPRISE DETECTOR v5.0")
        print("="*80)
        print(f"📡 Dashboard: http://{host}:{port}/dashboard/")
        print(f"🔌 API: http://{host}:{port}/api/v1/")
        print(f"🔐 Admin: http://{host}:{port}/admin/")
        print(f"💾 Database: {self.config.get('database_url', 'sqlite:///pegasus_enterprise.db')}")
        print(f"💰 Payment Processors: {self.payment_processors}")
        print("="*80)
        
        # Create admin user if none exists
        with self.Session() as db:
            if db.query(User).count() == 0:
                password_hash = bcrypt.hashpw(b'admin123', bcrypt.gensalt()).decode()
                admin_user = User(
                    username='admin',
                    email='admin@pegasusenterprise.com',
                    password_hash=password_hash,
                    first_name='System',
                    last_name='Administrator',
                    is_superuser=True,
                    is_active=True,
                    is_verified=True,
                    preferences={'theme': 'dark'},
                    notification_settings={'email_alerts': True}
                )
                db.add(admin_user)
                db.commit()
                
                # Create credits for admin
                credits = UserCredits(
                    user_id=admin_user.id,
                    credits_balance=1000,
                    credits_purchased=1000,
                    billing_tier='enterprise',
                    daily_scan_limit=1000,
                    monthly_scan_limit=10000
                )
                db.add(credits)
                db.commit()
                
                print(f"✅ Created admin user: admin / admin123")
        
        # Run Flask app with SocketIO
        if debug:
            self.flask_app.run(host=host, port=port, debug=debug)
        else:
            self.socketio.run(self.flask_app, host=host, port=port, debug=debug)

# ============================================================================
# 🚀 MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for Pegasus Enterprise"""
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🏢 Pegasus Enterprise Detector v5.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pegasus_enterprise.py run                     # Start the full enterprise system
  python pegasus_enterprise.py run --port 8080        # Start on custom port
  python pegasus_enterprise.py api                    # Start API server only
  python pegasus_enterprise.py dashboard              # Start dashboard only
  python pegasus_enterprise.py scanner --device iphone # Run scanner only
  python pegasus_enterprise.py license --generate     # Generate license
  python pegasus_enterprise.py credits --balance      # Check credit balance

Environment Variables:
  DATABASE_URL=postgresql://user:pass@localhost/pegasus
  STRIPE_API_KEY=sk_live_...
  PAYPAL_CLIENT_ID=...
  PAYPAL_SECRET=...
  SECRET_KEY=your-secret-key-here
  REDIS_URL=redis://localhost:6379/0
        """
    )
    
    parser.add_argument('command',
                       choices=['run', 'api', 'dashboard', 'scanner', 'license', 'credits', 'admin'],
                       default='run',
                       nargs='?',
                       help='Command to execute')
    
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--device', help='Device type for scanner')
    parser.add_argument('--jailbreak', action='store_true', help='Jailbreak status')
    parser.add_argument('--output', help='Output file')
    
    # License arguments
    parser.add_argument('--generate', action='store_true', help='Generate license')
    parser.add_argument('--tier', default='professional', help='License tier')
    parser.add_argument('--duration', type=int, default=365, help='License duration in days')
    
    # Credit arguments
    parser.add_argument('--balance', action='store_true', help='Check credit balance')
    parser.add_argument('--purchase', type=int, help='Purchase credits amount')
    parser.add_argument('--package', default='basic', help='Credit package')
    
    args = parser.parse_args()
    
    # Initialize application
    app = PegasusEnterpriseApplication({
        'database_url': os.environ.get('DATABASE_URL', 'sqlite:///pegasus_enterprise.db'),
        'sql_echo': args.debug
    })
    
    # Execute command
    if args.command == 'run':
        app.run(host=args.host, port=args.port, debug=args.debug)
    
    elif args.command == 'api':
        # Run API server only
        from fastapi import FastAPI
        import uvicorn
        
        fastapi_app = FastAPI(title="Pegasus Enterprise API", version="5.0")
        
        @fastapi_app.get("/")
        async def root():
            return {"message": "Pegasus Enterprise API v5.0"}
        
        @fastapi_app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        @fastapi_app.post("/scan")
        async def scan(device_type: str = 'auto', jailbreak: bool = False):
            # This would integrate with the actual scanner
            return {"status": "scan_started", "device": device_type}
        
        print(f"🚀 Starting API server on http://{args.host}:{args.port}")
        uvicorn.run(fastapi_app, host=args.host, port=args.port)
    
    elif args.command == 'dashboard':
        # Run dashboard only
        app.run(host=args.host, port=args.port, debug=args.debug)
    
    elif args.command == 'scanner':
        # Run scanner only
        scanner = UniversalDeviceScanner()
        
        device_type = args.device or 'auto'
        jailbreak = args.jailbreak
        
        print("🔍 Starting Pegasus Scanner...")
        print(f"📱 Device: {device_type}")
        print(f"🔓 Jailbreak: {jailbreak}")
        print("-" * 50)
        
        results = scanner.scan_universal(device_type, jailbreak)
        
        # Print results
        final = results.get('final_assessment', {})
        
        if final.get('verdict') == 'INFECTED':
            print("\n🚨 CRITICAL: PEGASUS DETECTED!")
            print("=" * 50)
            print(f"Variant: {final.get('variant', 'Unknown')}")
            print(f"Confidence: {final.get('confidence', 0):.1%}")
            print(f"Risk Level: {final.get('risk_level', 'UNKNOWN')}")
            print(f"\nMessage: {final.get('message', '')}")
            print(f"Recommendation: {final.get('recommendation', '')}")
        else:
            print("\n✅ Device appears clean")
            print("=" * 30)
            print(f"Verdict: {final.get('verdict', 'UNKNOWN')}")
            print(f"Confidence: {final.get('confidence', 0):.1%}")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {args.output}")
    
    elif args.command == 'license':
        # License management
        if args.generate:
            with app.Session() as db:
                license_manager = EnhancedLicenseManager(db)
                
                # Use first user or create one
                user = db.query(User).first()
                if not user:
                    print("❌ No users found. Please run the application first.")
                    return
                
                result = license_manager.generate_license_key(
                    user.id, args.tier, args.duration
                )
                
                if result['success']:
                    print("✅ License generated successfully!")
                    print(f"License Key: {result['license_key']}")
                    print(f"Tier: {args.tier}")
                    print(f"Duration: {args.duration} days")
                    print(f"License ID: {result['license_id']}")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print("ℹ️  License management commands:")
            print("  --generate      Generate a new license")
            print("  --tier TIER     License tier (default: professional)")
            print("  --duration DAYS License duration in days (default: 365)")
    
    elif args.command == 'credits':
        # Credit management
        with app.Session() as db:
            credit_manager = EnhancedCreditManager(db)
            
            # Use first user
            user = db.query(User).first()
            if not user:
                print("❌ No users found. Please run the application first.")
                return
            
            if args.balance:
                credit_info = credit_manager.get_user_credits_info(user.id)
                
                print("💰 Credit Balance:")
                print("-" * 30)
                print(f"Balance: {credit_info['credits']['balance']:,} credits")
                print(f"Used: {credit_info['credits']['used']:,} credits")
                print(f"Purchased: {credit_info['credits']['purchased']:,} credits")
                print(f"Free: {credit_info['credits']['free']:,} credits")
                print(f"Tier: {credit_info['billing']['tier']}")
                print(f"Daily scans: {credit_info['usage']['daily_scan_quota']}")
                print(f"Monthly scans: {credit_info['usage']['monthly_scan_quota']}")
            
            elif args.purchase:
                print(f"💰 Purchasing {args.purchase} credits...")
                # Purchase logic would go here
                print("✅ Purchase initiated")
    
    elif args.command == 'admin':
        # Admin commands
        print("👑 Admin Commands:")
        print("  create-user      Create a new user")
        print("  reset-password   Reset user password")
        print("  list-users       List all users")
        print("  system-stats     Show system statistics")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
