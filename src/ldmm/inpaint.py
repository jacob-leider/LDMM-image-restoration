import ldmm
from ldmm.patch_set import *
from ldmm.utils import *
from ldmm.sample_generation import *
from ldmm.weight_matrix_generation import *
import numpy
from zipfile import ZipFile
import os
from numpy.lib.stride_tricks import sliding_window_view
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from PIL import Image
from numpy.fft import fft2
import os
import time
import shutil


def LDMM_debug(y, omega,
         mu,
         patch_size=10, overlap=1,
         max_iter=1000,
         max_breg_iters = 20,
         tol=1e-6,
         gt=None,
         nn_algo='naive', # naive, KD-Tree, Ball-Tree, Annoy, ...
         randomize_unknown_pix=True,
         pad_initial=True,
         plot_freqs=False,
         plot_patches=True,
         return_with_padding=False,
         iters_before_plotting=5,
         download_results=True,
         notes=None):
    """
    Implements LDMM to restore degraded image

    Args:
        y (numpy.ndarray): The degraded image
        omega (numpy.ndarray): The known region of the degraded image
        mu (float): parameter
        patch_size (int): The size of the patch set
        overlap (int): The amount of overlap when created the index set
        max_iter (int): The maximum number of iterations to run

    Returns:
        numpy.ndarray: The reconstructed image
    """

    # Open a directory to report results:
    os.mkdir("ldmm_results")
    writer = open("ldmm_results/parameters.txt", 'w')
    writer.write(f"""mu = {mu}
                 patch_size = {patch_size}
                 overlap = {overlap}
                 max_iter = {max_iter}
                 tol = {tol}
                 randomize_unknown_pix = {randomize_unknown_pix}
                 pad_initial = {pad_initial}
                 plot_freqs = {plot_freqs}
                 plot_patches = {plot_patches}
                 return_with_padding = {return_with_padding}
                 iters_before_plotting = {iters_before_plotting}
                 download_results = {download_results}
                 nn_algo = {nn_algo}
                 """)
    writer.close()

    os.mkdir('ldmm_results/image')
    os.mkdir('ldmm_results/patch')

    psnr_writer = open(f'ldmm_results/psnr.csv', 'a')
    psnr_writer.write('iter, psnr\n')
    psnr_writer.close()

    bench_writer = open(f'ldmm_results/bench.csv', 'a')
    bench_writer.write('enforce boundary, patch set construction, graph laplacian construction, coordinate update, image update')
    bench_writer.close()

    plot_psnrs = (gt.all() != None)
    psnrs = []
           
    # Local PSNR calculator function.
    def psnr(gt_f, f):
      return 10 * numpy.log10(1 / numpy.mean((gt_f - f) ** 2))
      
    # A map to nearest-neighbors algorithms from their names.
    nn_algo_map = {
        'naive': sparse_L_kd_tree, # TODO
        'kdtree': sparse_L_kd_tree,
        'balltree': sparse_L_ball_tree,
        'annoy': sparse_L_annoy
    }

    # Select nearest-neighbors algorithm.
    selected_nn_algo = None
    try:
      selected_nn_algo = nn_algo_map[nn_algo]
    except:
      print(f"Unknown nearest-neighbors algorithm: {nn_algo}")
      return

    # Pad the input (maybe).
    if pad_initial:
      y_padded = pad_image(y, patch_size, overlap)
      omega = pad_image(omega, patch_size, overlap)
    else:
      y_padded = y

    # Dimensions etc.
    m, n = y_padded.shape
    stride = patch_size - overlap

    # Find uniform index set to be used for the patch set (mostly for testing).
    index_set = uniform_index_set(m, n, patch_size, overlap)
    N = index_set.shape[0]
    unpadded_pos = (slice(stride, stride + y.shape[0]),
                    slice(stride, stride + y.shape[1]))

    # Precompute (P^*P)^{-1}
    p_star_p_inv = patch_set_contribution_mask(index_set, 
                                               (patch_size, patch_size), 
                                               y_padded.shape)
    p_star_p_inv[p_star_p_inv == 0] = 1
    p_star_p_inv = 1 / p_star_p_inv

    # Coordinate functions on the manifold.
    U = numpy.zeros((N, patch_size**2))

    d_n = numpy.zeros((index_set.shape[0], patch_size**2))

    # Initial guess for the image and bregman parameter.
    if randomize_unknown_pix:
      f_n = fill_degraded_pixels(y_padded, omega)
    else:
      f_n = y_padded

    plt.imshow(f_n, cmap='gray')
    plt.show()

    ############################################################################
    # MAIN LOOP                                                                #
    ############################################################################
    for iter in range(max_iter):
        psnr_writer = open(f'ldmm_results/psnr.txt', 'a')
        bench_writer = open(f'ldmm_results/bench.csv', 'a')
        print(f"{iter+1}/{max_iter}")

        # Enforce the reflective boundary condition so partials are zero w.r.t
        # boundary normals.
        t1 = time.time()
        f_n = enforce_boundary(f_n, patch_size, overlap)
        t2 = time.time()
        bench_writer.write(f"{t2 - t1}, ")
        # print(f"\tEnforce boundary: {t2 - t1}s")

        t1 = time.time()
        # Save copy of previous iteration
        f_prev = f_n.copy()
        P_f_n = patch_set(f_n, patch_size, index_set)
        t2 = time.time()
        bench_writer.write(f"{t2 - t1}, ")
        # print(f"\tPatch-set construction: {t2 - t1}s")

        ########################################################################
        # Compute weight matrices                                              #
        #                                                                      #
        #     W and W_bar, L                                                   #
        ########################################################################
        t1 = time.time()
        W, L = sparse_L_ball_tree(P_f_n, k_neighbors=60, sigma_neighbors=20)
        t2 = time.time()
        bench_writer.write(f"{t2 - t1}, ")
        # print(f"\tW, L construction: {t2 - t1}s")

        # ZERO OUT D TO SOLVE FOR U, f (maybe wrong)
        d_n = numpy.zeros((index_set.shape[0], patch_size**2))

        ########################################################################
        # INNER BREGMAN LOOP                                                            #
        ########################################################################
        for breg_iter in range(max_breg_iters):
          t1 = time.time()
          V = P_f_n - d_n
          t2 = time.time()
          if breg_iter == 3:
            bench_writer.write(f"{t2 - t1}, ")
          # print(f"\tV construction: {t2 - t1}s")

          ########################################################################
          # Coordinate function update                                           #
          #                                                                      #
          # Solve AU = B, where A = L + ¯μ ¯W, B = ¯μ ¯WV                        #
          ########################################################################
          # Precompute the matrix A = L + mu * W_bar
          t1 = time.time()
          A = L + mu * W
          B = mu * W.dot(V)

          # Solve the linear system for each column of V (each patch dimension)
          for i in range(patch_size**2):
              U[:, i], exit_code = gmres(A, B[:,i])
              if exit_code != 0:
                  print(f"GMRES did not converge for patch element {i}. \
                          Exit code: {exit_code}")

          t2 = time.time()
          if breg_iter == 3:
            bench_writer.write(f"{t2 - t1}, ")
          # print(f"\tCoordinate function update: {t2 - t1}s")

          ########################################################################
          # f-update                                                             #
          #                                                                      #
          # f^{n+1}(x) = f(x),                        x ∈ Ω,                     #
          #              (P^*P)^(-1)(P^*(U + d^n)),   x ∉ Ω                      #
          ########################################################################
          t1 = time.time()

          patch_set_adjoint_operator(U + d_n, index_set, (patch_size, patch_size), f_n)
          f_n = f_n * p_star_p_inv
          f_n[omega == 1] = y_padded[omega == 1]

          # Update d_n
          d_n = d_n + U - patch_set(f_n, patch_size, index_set)
          t2 = time.time()
          if breg_iter == 3:
            bench_writer.write(f"{t2 - t1}\n")
          bench_writer.write(f"{t2 - t1}\n")
          # print(f"\tf-update: {t2 - t1}s")

        # Convergence check.
        err = numpy.linalg.norm(f_n - f_prev, 'fro') / numpy.linalg.norm(f_prev, 'fro')
        print(f"Error: {err}")
        if err < tol:
            print(f"Converged after {iter + 1} iterations!")
            break

        psnrs.append(psnr(gt, f_n[unpadded_pos]))
        psnr_writer.write(f"{iter}, {psnrs[-1]}\n")

        # Track progress.
        if (iter+1) % iters_before_plotting == 0:
          image_arr = f_n[unpadded_pos]
          image_arr = numpy.clip(image_arr, 0, 1)
          image_arr = (image_arr * 255).astype(numpy.uint8)

          plt.figure(figsize=(10, 10))
          plt.imshow(image_arr, cmap='gray')
          plt.title(f"Iteration {iter+1}")
          plt.show()

          # Save the grayscale image.
          image = Image.fromarray(image_arr, 'L')
          image.save(f'ldmm_results/image/iter_{iter}.png')

          if plot_psnrs:
            plt.plot(psnrs)
            plt.title("PSNR")
            plt.show()

          if plot_freqs:
            plt.subplot(1, 2, 1)
            # Trace mean and std deviation of iterates (time domain).
            plt.hist(image_arr.ravel(), bins=200, range=[0, 1])
            plt.title("Time Domain")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            # Trace mean and std deviation of iterates (frequency domain).
            plt.subplot(1, 2, 2)
            plt.hist(fft2(image_arr).ravel(), bins=200, range=[-50, 50])
            plt.title("Frequency Domain")
            plt.xlabel("Frequency Value")
            plt.ylabel("Frequency")
            plt.show()

          if plot_patches:
            # Show selected patches.
            fig, ax = plt.subplots(1, 5, figsize=(18, 6))
            img = numpy.repeat(image_arr[:,:,numpy.newaxis], 3, axis=2)
            print(img.shape)
            # print(img.shape)
            # img[:,:,:] = image_arr[:,:,numpy.newaxis]
            for i in range(4):
              patch_idx = (i + 1) * (N // 6)
              patch_slice = (
                  slice(index_set[patch_idx][0],
                        index_set[patch_idx][0] + patch_size),
                  slice(index_set[patch_idx][1],
                        index_set[patch_idx ][1] + patch_size))
              # Show patch.
              ax[i + 1].set_title(f"Patch {patch_idx}")
              ax[i + 1].imshow(
                  numpy.reshape(P_f_n[patch_idx], (patch_size, patch_size)),
                  cmap='gray')
              # Hilight patch.
              img[patch_slice] = numpy.array([255, 0, 0])
            # Show current image with selected patches hilighted in red.
            ax[0].imshow(img)
            ax[0].set_title("Original Image")
            # Save to
            plt.savefig(f'ldmm_results/patch/iter_{iter}_patches.png')
            plt.show()

        bench_writer.close()
        psnr_writer.close()

    if return_with_padding:
      # Get rid of padding.
      f_n = f_n[stride:m - stride, stride:n-stride]

    if download_results:
      # Compress and download stats.
      shutil.make_archive('~/ldmm_results', 'zip', './ldmm_results') # TODO

    return f_n, psnrs


def LDMM(y, omega,
         mu,
         patch_size=10, overlap=1,
         max_iter=1000,
         max_breg_iters = 20,
         tol=1e-6,
         gt=None,
         nn_algo='naive', # naive, KD-Tree, Ball-Tree, Annoy, ...
         randomize_unknown_pix=True,
         pad_initial=True,
         plot_freqs=False,
         plot_patches=True,
         return_with_padding=False,
         iters_before_plotting=5,
         download_results=True,
         notes=None):
    """
    Implements LDMM to restore degraded image

    Args:
        y (numpy.ndarray): The degraded image
        omega (numpy.ndarray): The known region of the degraded image
        mu (float): parameter
        patch_size (int): The size of the patch set
        overlap (int): The amount of overlap when created the index set
        max_iter (int): The maximum number of iterations to run

    Returns:
        numpy.ndarray: The reconstructed image
    """

    # A map to nearest-neighbors algorithms from their names.
    nn_algo_map = {
        'naive': sparse_L_kd_tree, # TODO
        'kdtree': sparse_L_kd_tree,
        'balltree': sparse_L_ball_tree,
        'annoy': sparse_L_annoy
    }

    # Select nearest-neighbors algorithm.
    selected_nn_algo = None
    try:
      selected_nn_algo = nn_algo_map[nn_algo]
    except:
      print(f"Unknown nearest-neighbors algorithm: {nn_algo}")
      return

    # Pad the input (maybe).
    if pad_initial:
      y_padded = pad_image(y, patch_size, overlap)
      omega = pad_image(omega, patch_size, overlap)
    else:
      y_padded = y

    # Dimensions etc.
    m, n = y_padded.shape
    stride = patch_size - overlap

    # Find uniform index set to be used for the patch set (mostly for testing).
    index_set = uniform_index_set(m, n, patch_size, overlap)
    N = index_set.shape[0]
    unpadded_pos = (slice(stride, stride + y.shape[0]),
                    slice(stride, stride + y.shape[1]))

    # Precompute (P^*P)^{-1}
    p_star_p_inv = p_star_p(m, n, patch_size, overlap)
    p_star_p_inv[p_star_p_inv == 0] = 1
    p_star_p_inv = 1 / p_star_p_inv

    # Coordinate functions on the manifold.
    U = numpy.zeros((N, patch_size**2))
    d_n = numpy.zeros((index_set.shape[0], patch_size**2))

    # Initial guess for the image and bregman parameter.
    if randomize_unknown_pix:
      f_n = fill_degraded_pixels(y_padded, omega)
    else:
      f_n = y_padded

    ############################################################################
    # MAIN LOOP                                                                #
    ############################################################################
    for iter in range(max_iter):
        # Enforce the reflective boundary condition so partials are zero w.r.t
        # boundary normals.
        f_n = enforce_boundary(f_n, patch_size, overlap)

        # Save copy of previous iteration
        f_prev = f_n.copy()
        P_f_n = patch_set(f_n, patch_size, index_set)

        ########################################################################
        # Compute weight matrices                                              #
        #                                                                      #
        #     W and W_bar, L                                                   #
        ########################################################################
        W, L = sparse_L_ball_tree(P_f_n, k_neighbors=60, sigma_neighbors=20)

        # ZERO OUT D TO SOLVE FOR U, f (maybe wrong)
        d_n = numpy.zeros((index_set.shape[0], patch_size**2))

        ########################################################################
        # INNER BREGMAN LOOP                                                            #
        ########################################################################
        for breg_iter in range(max_breg_iters):
          V = P_f_n - d_n

          ########################################################################
          # Coordinate function update                                           #
          #                                                                      #
          # Solve AU = B, where A = L + ¯μ ¯W, B = ¯μ ¯WV                        #
          ########################################################################
          L_plus_mu_W = L + mu * W
          mu_W_V = mu * W.dot(V)
          # Solve the linear system for each column of V (each patch dimension)
          for i in range(patch_size**2):
              U[:, i], exit_code = gmres(L_plus_mu_W, mu_W_V[:,i])
              if exit_code != 0:
                  print(f"GMRES did not converge for patch element {i}. \
                          Exit code: {exit_code}")

          ########################################################################
          # f-update                                                             #
          #                                                                      #
          # f^{n+1}(x) = f(x),                        x ∈ Ω,                     #
          #              (P^*P)^(-1)(P^*(U + d^n)),   x ∉ Ω                      #
          ########################################################################

          patch_set_adjoint_operator(U + d_n, index_set, (patch_size, patch_size), f_n)
          f_n = f_n * p_star_p_inv
          f_n[omega == 1] = y_padded[omega == 1]

          # Update d_n
          d_n = d_n + U - patch_set(f_n, patch_size, index_set)

        # Convergence check.
        err = numpy.linalg.norm(f_n - f_prev, 'fro') / numpy.linalg.norm(f_prev, 'fro')
        if err < tol:
            print(f"Converged after {iter + 1} iterations!")
            break

    if return_with_padding:
      # Get rid of padding.
      f_n = f_n[stride:m - stride, stride:n-stride]

    return f_n
