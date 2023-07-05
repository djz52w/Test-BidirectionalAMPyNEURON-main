from neuron import h
import numpy as np

class Cell:
    def __init__(self, gid, x, y, z, theta, offset, ring_flag):
        self._gid = gid
        self.rangen = np.random
        self.rangen.seed(gid)
        self._setup_morphology()
        self.all = self.soma.wholetree()
        self._setup_biophysics()
        self.x = self.y = self.z = 0
        h.define_shape()
        if ring_flag:
            self._rotate_z(z, theta)
        self._set_position(0, 0, 0, offset)
        self.coord = [x, y, z, theta, offset]
        # Record spike times
        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)

        self._ncs = []

    def __repr__(self):
        return '{}[{}]'.format(self.name, self._gid)

    def _set_position(self, x, y, z, offset):
        for sec in self.all:
            for i in range(sec.n3d()):
                #znew = z - self.z + sec.z3d(i) + offset[2]
                #print(znew)
                sec.pt3dchange(i,
                               x - self.x + sec.x3d(i) + offset[0],
                               y - self.y + sec.y3d(i) + offset[1],
                               z - self.z + sec.z3d(i) + offset[2],
                              sec.diam3d(i))
        self.x, self.y, self.z = x, y, z

    def _rotate_z(self, z, theta):
        """Rotate the cell about the Z axis."""
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                sec.pt3dchange(i, xprime, yprime, z, sec.diam3d(i))

    def _get_data(self):
        start_ends = []
        radii = []
        seg_order = []
        for sec in self.all:
            ds = 1e-3
            boundary_pos = np.linspace(ds,1-ds,sec.nseg+1)
            segs = [str(seg) for seg in sec]
            for jj in range(sec.nseg):
                start_pos = boundary_pos[jj]
                end_pos = boundary_pos[jj+1]
                for ii in range(int(h.n3d(sec=sec))-1):
                    if (h.arc3d(ii, sec=sec)/sec.L) <= start_pos <= (h.arc3d(ii+1, sec=sec)/sec.L):
                        swc1 = np.array([h.x3d(ii, sec=sec),h.y3d(ii, sec=sec),h.z3d(ii, sec=sec)])
                        swc2 = np.array([h.x3d(ii+1, sec=sec),h.y3d(ii+1, sec=sec),h.z3d(ii+1, sec=sec)])
                        f = (start_pos-h.arc3d(ii, sec=sec)/sec.L)/((h.arc3d(ii+1, sec=sec)-h.arc3d(ii, sec=sec))/sec.L)
                        break

                start_point = tuple(f*(swc2-swc1)+swc1)

                for ii in range(int(h.n3d(sec=sec))-1):
                    if (h.arc3d(ii, sec=sec)/sec.L) <= end_pos <= (h.arc3d(ii+1, sec=sec)/sec.L):
                        swc1 = np.array([h.x3d(ii, sec=sec),h.y3d(ii, sec=sec),h.z3d(ii, sec=sec)])
                        swc2 = np.array([h.x3d(ii+1, sec=sec),h.y3d(ii+1, sec=sec),h.z3d(ii+1, sec=sec)])
                        f = (end_pos-h.arc3d(ii, sec=sec)/sec.L)/((h.arc3d(ii+1, sec=sec)-h.arc3d(ii, sec=sec))/sec.L)
                        break

                end_point = tuple(f*(swc2-swc1)+swc1)

                start_ends.append([start_point,end_point])
                radii.append([sec(boundary_pos[jj]).diam/2,sec(boundary_pos[jj+1]).diam/2])

                seg_key = segs[jj].split('.')
                seg_key = seg_key[1]+'.'+seg_key[2]
                seg_order.append(seg_key)

        return start_ends, radii, seg_order

    def _connect_dots(self, data, diams, num_pts, offset_global):
        x = []
        y = []
        z = []
        connections = []
        offset = 0
        for kk in range(len(data)):
            # Define points
            C1 = np.array(data[kk][0])
            C2 = np.array(data[kk][1])
            # Define normal plane
            p = C1-C2
            d = np.dot(p,C1)
            # Get normal vectors on plane
            z_idx = np.arange(3)[p==0]
            nz_idx = np.arange(3)[p!=0]
            if len(nz_idx) == 3:
                x1 = 1.
                y1 = 1.
                z1 = (d-(np.dot(p[:2],[x1,y1])))/p[2]
                a = np.array([x1,y1,z1])
            elif len(nz_idx) == 2:
                a = np.zeros(3)
                a[z_idx] = 1.
                a[nz_idx[0]] = 1.
                a[nz_idx[1]] = (d-p[nz_idx[0]])/p[nz_idx[1]]
                #x1 = 1.
                #y1 = (d-p[0])/p[1]
                #z1 = 1.
            else:
                a = np.zeros(3)
                a[z_idx] = 1.
                a[nz_idx] = d/p[nz_idx]
                #x1 = d/p[0]
                #y1 = 1.
                #z1 = 1.
            a = a-C1
            if len(p[p!=0]) == 3:
                x2 = 1.
                y2 = (a[2]*p[0]/p[2] - a[0]) / (a[1] - a[2]*p[1]/p[2])
                z2 = -(p[1]*y2+p[0])/p[2]
                b = np.array([x2,y2,z2])
            elif len(p[p!=0]) == 2:
                b = np.zeros(3)
                b[z_idx] = 1.
                b[nz_idx[0]] = a[z_idx]/(a[nz_idx[1]]*p[nz_idx[0]]/p[nz_idx[1]] - a[nz_idx[0]])
                b[nz_idx[1]] = -p[nz_idx[0]]*b[nz_idx[0]]/p[nz_idx[1]]
                #x2 = a[2]/(a[1]*p[0]/p[1] - a[0])
                #y2 = -p[0]*x2/p[1]
                #z2 = 1.
            else:
                b = np.zeros(3)
                b[nz_idx] = 0
                b[z_idx[0]] = 1.
                b[z_idx[1]] = -a[z_idx[0]]/a[z_idx[1]]
                #x2 = 0
                #y2 = 1.
                #z2 = -a[1]/a[2]

            # Convert to unit vectors
            a = a/np.linalg.norm(a)
            b = b/np.linalg.norm(b)
            theta_step = np.pi*2/num_pts
            # Define set of points at a defined radius around
            # the original points, C1 and C2
            P1 = np.zeros((num_pts,3))
            P2 = np.zeros((num_pts,3))
            r1 = diams[kk][0]
            r2 = diams[kk][1]
            theta = 0
            for ii in range(num_pts):
                for jj in range(3):
                    P1[ii][jj] = C1[jj] + r1*np.cos(theta)*a[jj] + r1*np.sin(theta)*b[jj]
                    P2[ii][jj] = C2[jj] + r2*np.cos(theta)*a[jj] + r2*np.sin(theta)*b[jj]

                theta += theta_step

            # Define triangles
            for ii in range(2*num_pts):
                if ii < num_pts:
                    connections.append((ii+offset+offset_global,(ii+1)%num_pts+offset+offset_global,ii+num_pts+offset+offset_global))
                else:
                    connections.append((ii+offset+offset_global,(ii+1-num_pts)%num_pts+offset+offset_global+num_pts,(ii-num_pts+1)%num_pts+offset+offset_global))

            for ii in range(num_pts):
                x.append(P1[ii][0])
                y.append(P1[ii][1])
                z.append(P1[ii][2])

            for ii in range(num_pts):
                x.append(P2[ii][0])
                y.append(P2[ii][1])
                z.append(P2[ii][2])

            offset += 2*num_pts

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        return x, y, z, connections
