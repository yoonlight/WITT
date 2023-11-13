from net.decoder import *
from net.encoder import *
from loss.distortion import Distortion
from net.channel import Channel
from random import choice
import torch.nn as nn
import scipy.io as sio

def get_RRC():
    # args.sps = 10, roll-off factor = 0.5
    RRCmat = sio.loadmat('RRC.mat')
    data = RRCmat['Pulse']
    RRC = torch.from_numpy(data).float()
    return RRC.unsqueeze(1), RRC.size(dim=-1)


class WITT(nn.Module):
    def __init__(self, args, config):
        super(WITT, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.downsample = config.downsample
        self.model = args.model
        self.RRC, self.lenRRC = get_RRC()
        self.N = 64
        self.M = 128
        self.len_cyclic_prefix = 16
        self.sample_per_symbol = 10
        # generate carrier
        fc = 25e6  # carrier frequency
        fb = 10e6  # baseband frequency
        fs = fb * self.sample_per_symbol  # sampling rate
        numSamples = int(256/self.N) * (self.M+self.len_cyclic_prefix) * \
            self.sample_per_symbol + self.lenRRC * 2  # number of samples
        tt = torch.arange(0, numSamples/fs, 1/fs)
        self.carrier_cos: torch.Tensor = np.sqrt(
            2) * torch.cos(2 * np.pi * fc * tt)
        self.carrier_sin: torch.Tensor = np.sqrt(
            2) * torch.sin(2 * np.pi * fc * tt)
        self.thres = 0

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, given_SNR = None):
        B, C, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR
        # Encoder
        feature = self.encoder(input_image, chan_param, self.model)

        # power norm
        feature = self.power_norm(feature)
        # reshape to a vector BS*256*2 (for construting complex symbols)
        feature = feature.view(B, 256, 2)
        # (512 real symbols) to (256 complex symbols); power of real = 1; power of complex = 2
        feature = torch.view_as_complex(feature)
        # modulation
        # total subcarriers = M = 128, # allocated subcarriers = N = 64
        num_ofdm = int(256/self.N)
        feature = feature.view(B, num_ofdm, self.N)
        # mapping to subcarriers
        # determine locations of subcarriers
        # OFDMA, random mapping
        loc = torch.randperm(self.M)[:self.N].cuda()
        # power of x = 2; power of data_f = 1
        data_f = torch.zeros(B, num_ofdm, self.M, dtype=torch.complex64).cuda()
        data_f[:, :, loc] = feature
        # IFFT; power of data_f = 1, power of data_t = 1
        data_t = torch.fft.ifft(data_f, n=None, dim=-1, norm="ortho")
        # add CP
        len_data_t = num_ofdm * (self.M + self.len_cyclic_prefix)
        if self.len_cyclic_prefix != 0:
            data_t = torch.cat(
                [data_t[:, :, -self.len_cyclic_prefix:], data_t], dim=-1)
        # reshape back to a packet
        data_t = data_t.view(B, len_data_t)
        # oversampling
        data_t_over = torch.zeros(B, len_data_t*self.sample_per_symbol, dtype=torch.complex64).cuda()
        data_t_over[:, np.arange(0, len_data_t*self.sample_per_symbol, self.sample_per_symbol)] = data_t
        # pulse shaping (real in, complex out)
        data_t_over = torch.view_as_real(data_t_over)
        data_x = self.pulse_filter_complex_sig(data_t_over).cuda()
        # RF signal (real)
        carrier_cos = self.carrier_cos.to(data_x.device)
        carrier_sin = self.carrier_sin.to(data_x.device)
        data_x = data_x.real * carrier_cos[:data_x.size(1)] \
            - data_x.imag * carrier_sin[:data_x.size(1)]
        # compute PAPR
        PAPRdB, PAPRloss = self.compute_PAPR(data_x, len_data_t)
        feature = data_x
        CBR = feature.numel() / 2 / input_image.numel()
        # Feature pass channel
        # if self.pass_channel:
        #     noisy_feature = self.feature_pass_channel(feature, chan_param)
        # else:
        #     noisy_feature = feature

        # Channel
        data_r = self.awgn_channel(data_x, chan_param)

        # demodulation
        # baseband signal
        carrier_cos = self.carrier_cos.to(data_r.device)
        carrier_sin = self.carrier_sin.to(data_r.device)
        data_r_real = data_r * carrier_cos[:data_r.size(1)]
        data_r_imag = data_r * -carrier_sin[:data_r.size(1)]
        data_r_cpx = torch.cat(
            [data_r_real.unsqueeze(2), data_r_imag.unsqueeze(2)], dim=2)
        # matched filtering (real in, complex out)
        data_r_filtered = self.pulse_filter_complex_sig(data_r_cpx)
        # synchronization and samling
        samplingloc = torch.arange(
            self.lenRRC-1, data_r_filtered.size(1)-self.lenRRC, self.sample_per_symbol)
        y = torch.index_select(
            data_r_filtered, 1, samplingloc.cuda())
        # remove CP
        y = y.view(B, num_ofdm, self.M+self.len_cyclic_prefix)
        y = y[:, :, self.len_cyclic_prefix:]
        # FFT
        y = torch.fft.fft(y, n=None, dim=-1, norm="ortho")
        # demapping
        y = torch.index_select(y, 2, loc)
        # reshape to a packet, BS*4*64 -> BS*256
        y = y.view(B, num_ofdm * self.N)
        # complex to real, BS*256*2
        y = torch.view_as_real(y)
        # reshape to a compressed image
        noisy_feature = y.view(y.size(0), 64, 8)
        # Decoder
        recon_image = self.decoder(noisy_feature, chan_param, self.model)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        loss = self.squared_difference(recon_image.clamp(0., 1.), input_image)
        MSE_each_image = (torch.sum(loss.view(B,-1),dim=1))/(C*H*W)
        PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
        one_batch_PSNR = PSNR_each_image.data.cpu().numpy()
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean(), PAPRdB, PAPRloss, one_batch_PSNR

    def power_norm(self, feature) -> torch.Tensor:
        in_shape = feature.shape
        sig_in = feature.reshape(in_shape[0], -1)
        # each pkt in a batch, compute the mean and var
        sig_mean = torch.mean(sig_in, dim=1)
        sig_std = torch.std(sig_in, dim=1)
        # normalize
        sig_out = (sig_in-sig_mean.unsqueeze(dim=1)) / \
            (sig_std.unsqueeze(dim=1)+1e-8)
        return sig_out.reshape(in_shape)

    def pulse_filter_complex_sig(self, x):
        xreal = F.conv1d(x[:, :, 0].unsqueeze(1), self.RRC.flip(dims=[2]).cuda(), stride=1, padding=self.lenRRC-1)
        ximag = F.conv1d(x[:, :, 1].unsqueeze(1), self.RRC.flip(dims=[2]).cuda(), stride=1, padding=self.lenRRC-1)
        # convert back to complex
        x = torch.cat([xreal.squeeze(1).unsqueeze(2), ximag.squeeze(1).unsqueeze(2)], dim=2)
        return torch.view_as_complex(x)

    def compute_PAPR(self, x_t, len_data_t):
        # truncation, only compute the PAPR of the signal part (keep the signal length to len_data_t)
        truncloc = torch.arange(
            int((self.lenRRC+1)/2), int(((self.lenRRC+1)/2+len_data_t*self.sample_per_symbol)))
        x_t = torch.index_select(x_t, 1, truncloc.cuda())
        # compute PAPR
        data_t_power = torch.square(torch.abs(x_t))
        mean_power = torch.mean(data_t_power, dim=1)
        max_power = torch.max(data_t_power, dim=1).values
        PAPRdB = 10 * torch.log10(max_power/mean_power)
        PAPRloss = F.relu(PAPRdB-self.thres).mean()
        return PAPRdB, PAPRloss

    def awgn_channel(self, data_x, snr):
        batch_size, len_data_x = data_x.size(0), data_x.size(1)
        noise_std = 10 ** (-snr * 1.0 / 10 / 2)
        # real channel
        AWGN = torch.normal(0, std=noise_std, size=(
            batch_size, len_data_x), requires_grad=False).cuda()

        hh = np.array([1])
        data_r = torch.from_numpy(hh).type(torch.float32).cuda() * data_x + AWGN
        return data_r
