{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is a notebook that looks at sequential learning with some toy cases  \n",
    "Author: Andrew Holmgren  \n",
    "BSD3 license"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import scipy\n",
    "import sklearn\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "normal = lambda x, u, s: 1 / np.sqrt(2 * np.pi * s**2) * np.exp(-(x - u) / (2 * s**2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Simple model with linear parameters"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## The model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "(3.7) in Bishop, \"Pattern Recognition and Machine Learning\"  \n",
    "$$\n",
    "t = y(\\mathbf{x}, \\mathbf{w}) + \\epsilon\n",
    "$$\n",
    "giving\n",
    "$$\n",
    "p(\\mathbf{t} \\mid \\mathbf{X}, \\mathbf{w}, \\beta) = \\pi_{n=1}^N \\mathcal{N}(t_N\\mid \\mathbf{w}^T \\phi(\\mathbf{x}_n), \\beta^{-1})\n",
    "$$\n",
    "where $\\phi$ are modeled basis functions such that\n",
    "$$\n",
    "y(\\mathbf{x}, \\mathbf{w})=\\sum_{j=0}^{M-1}w_j \\phi_j(\\mathbf{x}) = \\mathbf{w}^T \\phi(\\mathbf{x})\n",
    "$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = lambda x, w: w[0] + w[1] * x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.3"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "w_true = [-0.3, 0.5]\n",
    "y(0, w_true)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "beta = (1 / .2)**2\n",
    "alpha = 2."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Generate the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0.54745344  0.93100045  0.63493354 -0.69220357 -0.99392913 -0.52278428\n",
      " -0.99281482  0.87394508  0.51413821 -0.85681813 -0.16657292 -0.02506078\n",
      "  0.38854114 -0.83446589  0.4818891   0.12262944  0.20868325  0.0241436\n",
      " -0.54247351  0.09946953]\n"
     ]
    }
   ],
   "source": [
    "x_vals = np.random.rand(20) * 2 - 1\n",
    "print(x_vals)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "t = y(x_vals, w_true) + np.random.normal(loc=0, scale=np.sqrt(1 / beta), size=x_vals.size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0xa1978aef0>,\n",
       " <matplotlib.lines.Line2D at 0xa197920b8>,\n",
       " <matplotlib.lines.Line2D at 0xa19792908>]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure()\n",
    "plt.plot(x_vals, t, '.b', x_vals[:2], t[:2], '.r', x_vals[:1], t[:1], 'xm')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Set up the posterior"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "v_m0 = np.array([0., 0.])\n",
    "m_S0 = 1/alpha*np.identity(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
