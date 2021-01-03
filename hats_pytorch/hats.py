import torch
from torch import nn
from torch.nn import functional as F
from hats_cuda import group_by_cell, local_surface


class HATS(nn.Module):

    def __init__(self, input_shape=(180, 240), r=3,
                 k=10, tau=1e6, delta_t=100000, fold=False):
        """
        Implements the HATS event representation as described in "HATS:
        Histograms of Averaged Time Surfaces for Robust Event-based Object
        Classification", Sironi et al.

        :param tuple input_shape: the (height, width) size of the frame
        :param int r: the size of the 2r+1 x 2r+1 spatial neighborhood
        :param int k: the size of the cells
        :param float tau: the decay factor
        :param float delta_t: the delay defining the temporal neighborhood
        :param bool fold: if True, histograms are organized in a grid of
            shape [B, 2, (2R+1) * NhCells, (2R+1) * NwCells] where each
            histogram is placed its original position in the frame. Otherwise,
            a [B, NCells, 2, 2R+1, 2R+1] tensor is returned
        """
        super().__init__()

        self.fold = fold
        self.delta_t = delta_t
        self.tau = tau
        self.r = r
        self.k = k

        # Make sure the input shape is divisible by k
        padded_shape = tuple((s + (k - 1)) // k * k for s in input_shape)
        self.input_shape = padded_shape
        height, width = padded_shape

        self.grid_n_width = (width // k)
        self.grid_n_height = (height // k)
        self.n_cells = self.grid_n_width * self.grid_n_height
        self.cell_size = 2*self.r + 1
        self.n_polarities = 2

        self.output_shape = (2, self.grid_n_height * self.cell_size,
                             self.grid_n_width * self.cell_size)

        self.register_buffer('coord2cellid', self.get_coord2cellid_matrix())

    def get_coord2cellid_matrix(self):

        flat_matrix = torch.arange(self.n_cells) \
            .repeat(self.k ** 2) \
            .view(1, -1, self.n_cells).float()
        pixels2rf_matrix = F.fold(flat_matrix, self.input_shape,
                                  kernel_size=(self.k, self.k), stride=self.k)
        return pixels2rf_matrix.type(torch.int64).view(self.input_shape)

    def group_by_cell(self, events, lengths):

        # Retrieve the cell id in which events are contained
        ids = self.coord2cellid[events[..., 1].type(torch.int64),
                                events[..., 0].type(torch.int64)]

        # Group events by cell id, maintaining the original event order
        # groups.shape = [n_ev, n_rf, n_feat]
        return group_by_cell(events, ids, lengths,
                             self.grid_n_height, self.grid_n_width)

    def local_surface(self, gr_events, gr_len):
        # [TCellPad, 2, 2R+1, 2R+1, BNCells]
        gr_local_surfaces = local_surface(gr_events, gr_len,
                                          self.delta_t, self.r, self.tau)
        # [TCellPad, 2, 2R+1, 2R+1, BNCells] -> [2, 2R+1, 2R+1, BNCells]
        #  -> [BNCells, 2, 2R+1, 2R+1]
        gr_local_surfaces = gr_local_surfaces.sum(0).permute(3, 0, 1, 2)
        return gr_local_surfaces

    def forward(self, events, lengths):
        """
        :param torch.Tensor events: the input events organized as a tensor of
            shape [B, TPad, C], where B is the batch dimension, TPad is the
            temporal dimension (padded at the end with zero values if sample b
            is shorter), and C is the channel dimension containing (x, y, t, p)
            features
        :param torch.Tensor lengths: a tensor of shape [B] specifying for each
            sample in "events", its original length before padding
        :return: if self.fold=True, histograms are organized in a grid of
            shape [B, 2, (2R+1) * NhCells, (2R+1) * NwCells] where each
            histogram is placed its original position in the frame. Otherwise,
            a [B, NCells, 2, 2R+1, 2R+1] tensor is returned
        """
        B, TPad, C = events.shape
        # Ensure that polarities are encoded as [0, 1] values
        assert events[..., -1].min().item() >= 0 and \
               events[..., -1].max().item() <= 1
        lengths = lengths.long()

        # gr_events.shape = [TCellPad, B*NCells, C]
        gr_b, gr_len, gr_h, gr_w, gr_events = self.group_by_cell(
            events, lengths)

        # Compute the local surfaces [BNCells, 2, 2R+1, 2R+1]
        gr_local_surfaces = self.local_surface(gr_events, gr_len)

        # Compute how many events there are in each cell
        norm_denom = gr_len[:, None].float() + 1e-6  # [B*NCells, 1]

        # Normalize by the event count
        cell_histograms = gr_local_surfaces / norm_denom[:, :, None, None]

        # Unfold the histograms back together
        histogram_unfold = cell_histograms.new_zeros(
            [B, self.grid_n_height, self.grid_n_width, 2,
             self.cell_size, self.cell_size])
        histogram_unfold[gr_b, gr_h, gr_w] = cell_histograms

        if not self.fold:
            return histogram_unfold.reshape(B, -1, 2, self.cell_size,
                                            self.cell_size)
        else:
            histogram_unfold = histogram_unfold.permute(0, 3, 4, 5, 1, 2) \
                .reshape(B, 2 * self.cell_size**2, self.n_cells)

            histogram = F.fold(histogram_unfold, self.output_shape[-2:],
                               kernel_size=self.cell_size,
                               stride=self.cell_size)
            return histogram
