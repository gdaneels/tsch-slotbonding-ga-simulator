import datetime

class Visualization:
    COLOR_EMPTY = 'D8D8D8'
    COLOR_NORMAL = 'b3ffb3'
    COLOR_OVERLAP = 'ff9999'
    # COLOR_MINIMAL = 'BEBEBE'

    def __init__(self, slots, frequencies, nds, parents, interferers):
        self.nr_slots = slots
        self.nr_frequencies = frequencies
        self.nodes = nds
        self.parents = parents
        self.reliability = {}
        self.parents[0] = 'NA'
        self.reliability[0] = 'NA'
        self.interferers = interferers
        # self.minimal_slots = minimal_slots
        self.nodes_sigma = {}
        self.schedule = {}
        self.available_slots = None # number of available slots
        self.obj_val = None

        for slot in range(slots):
            for frequency in range(frequencies):
                self.schedule[slot, frequency] = []

    def add_available_slots(self, sl):
        self.available_slots = sl

    def add_obj_val(self, obj_val):
        self.obj_val = obj_val

    def add_sigma(self, t, f, n, s):
        for _t in range(t, t + s):
            self.schedule[_t, f].append((t, f, n, s))

        if n not in self.nodes_sigma:
            self.nodes_sigma[n] = []
        self.nodes_sigma[n].append((t, f, n, s))

    def add_node(self, t, f, n, s):
        self.schedule[t, f].append((t, f, n, s))

    def visualize(self, suffix, output_dir):
        output = '<html><head><title>ADAPTSCH ILP solution</title></head><body style="font-family:arial;">'

        ### show schedule
        output += '<h1>ILP output at: %s</h1>' % str(datetime.datetime.now())
        output += '<h1>Schedule</h1><table border="0" cellpadding="5" style="font-size:11px;">'
        output += '<tr><th width="100" height="25" bgcolor="#F0F0F0">/</th>'
        for slot in range(self.nr_slots):
            color = 'F0F0F0'
            # if slot in self.minimal_slots:
            #     color = self.COLOR_MINIMAL
            output += '<th width="100" height="25" bgcolor="#%s">%s</th>' % (color, slot)
        output += '</tr>'
        for frequency in range(self.nr_frequencies):
            output += '<tr><th width="100" height="25" bgcolor="#F0F0F0">%s</th>' % (frequency)
            for slot in range(self.nr_slots):
                data = ''
                for (t, f, n, s) in self.schedule[slot, frequency]:
                    data += '&sigma;(n: %s, t: %s, f: %s, s: %s)</br>' % (n, t, f, s)
                color = self.COLOR_EMPTY
                if len(self.schedule[slot, frequency]) > 1:
                    color = self.COLOR_OVERLAP
                elif len(self.schedule[slot, frequency]) == 1:
                    color = self.COLOR_NORMAL
                output += '<td width="100" height="25" bgcolor="#%s"><center>%s</center></td>' % (color, data)
            output += '</tr>'
        output += '</table></font>'

        ### show node

        # output += '<h1>Decision variables</h1><table border="0" cellpadding="5" style="font-size:11px;">'
        # output += '<tr><th width="100" height="25" bgcolor="#F0F0F0">Node</th><th width="100" height="25" bgcolor="#F0F0F0">Parent</th><th width="100" height="25" bgcolor="#F0F0F0">&sigma;(t,f,n,s)</th><th width="100" height="25" bgcolor="#F0F0F0">interferers</th></tr>'
        # for n in self.nodes:
        #     output += '<tr>'
        #     output += '<td width="100" height="25" bgcolor="#F0F0F0"><center>%s</center></td>' % (n)
        #     output += '<td width="100" height="25" bgcolor="#F0F0F0"><center>%s</center></td>' % (self.parents[n])
        #     tmp_nodes_sigma = '/'
        #     if n in self.nodes_sigma:
        #         tmp_nodes_sigma = ''
        #         for s in self.nodes_sigma[n]:
        #             tmp_nodes_sigma += '%s</br>' % (str(s))
        #     output += '<td width="100" height="25" bgcolor="#F0F0F0"><center>%s</center></td>' % (tmp_nodes_sigma)
        #     output += '<td width="100" height="25" bgcolor="#F0F0F0"><center>%s</center></td>' % (" ")
        #     output += '</tr>'
        # output += '</table></font>'

        output += '</body></html>'

        name = '{0}/visualization-{1}.html'.format(output_dir, suffix)
        with open(name, "w") as html_file:
            html_file.write(output)

def main():
    slotframe_size = 11
    nr_frequencies = 16
    nodes = [0, 1, 2, 3]
    viz = Visualization(slotframe_size, nr_frequencies, nodes)

    viz.add_sigma(5, 2, 1, 3)
    viz.add_sigma(7, 2, 1, 3)
    viz.add_beta(2, 'QPSK_FEC_1_2', 1)
    viz.visualize()

if __name__ == "__main__":
    main()