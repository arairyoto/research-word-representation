my $syns1 = $ARGV[0];
my $syns2 = $ARGV[1];
#my $fname = $ARGV[2];

# for all syntactic category
use WordNet::QueryData;
use WordNet::Similarity::vector;
#use WordNet::Similarity::lesk;

use open OUT => ":utf8";

my $wn = WordNet::QueryData->new;
my $measure_vec = WordNet::Similarity::vector->new ($wn);
#my $measure_lesk = WordNet::Similarity::lesk->new ($wn);
my @s1v = split /,/,$syns1;
my @s2v = split /,/,$syns2;

if($#s1v!=$#s2v) {
	print "ERROR: number of synsets must be same";
	exit 1;
}

# $fname = "syn_dist.csv";
#open(FH, "> $fname") or die "$!";

#print FH "synset1,synset2,vector\n";
for (my $i=0; $i <= $#s1v; $i++){
	$s1 = $s1v[$i];
	$s2 = $s2v[$i];
	# Vector distance
	my $value_vec = $measure_vec->getRelatedness($s1, $s2);
	my ($error_vec, $errorString_vec) = $measure_vec->getError();
	die $errorString_vec if $error_vec;
	# Lesk
    #my $value_lesk = $measure_lesk->getRelatedness($s1, $s2);
    #my ($error_lesl, $errorString_lesk) = $measure_lesk->getError();
    #die $errorString_lesk if $error_lesk;
	
	print "$value_vec\n";
}
#close(FN);

# For DEBUG

#my $syns1 = "active#a#05,active#a#05,active#a#05,active#a#14,active#a#14,active#a#14,active#a#07,active#a#07,active#a#07,active#a#01,active#a#01,active#a#01,active#a#03,active#a#03,active#a#03,active#a#13,active#a#13,active#a#13,active#a#12,active#a#12,active#a#12,active#a#11,active#a#11,active#a#11,active#a#10,active#a#10,active#a#10,active#a#06,active#a#06,active#a#06,active#a#09,active#a#09,active#a#09,active#a#08,active#a#08,active#a#08,active#a#04,active#a#04,active#a#04,active#a#02,active#a#02,active#a#02";
#my $syns2 = "passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02,passive#a#01,passive#a#03,passive#a#02";
#$fname = "test.csv";
