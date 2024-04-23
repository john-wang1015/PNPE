#include "Model.h"

Pancreas* SeedAndGrowToStartVolume(double p0, double psc, int dmax, int gage, int page, double startVolume)
{
	Params* parameters = new Params(p0, psc, dmax, gage, page);
	vector<Cell*> empty;
	Pancreas* pancreas = new Pancreas(empty, parameters);
	// start with just one infected cancer cell nearest to (0, 0)
	pancreas->CreateInitialTumour();

	// pre-observation phase - run until tumour reaches start volume
	double volume = 0;
	int days = 0;
	while (volume < startVolume && days < 200)
		volume = pancreas->SimulateOneDay(days++);

	// who disposes parameters???
	return pancreas;
}

Pancreas* CreateNewParticle(double p0, double psc, int dmax, int gage, int page, Pancreas* pancreas)
{
	return pancreas->CreateNewParticle(new Params(p0, psc, dmax, gage, page));
}

void UpdateParticle(double p0, double psc, int dmax, int gage, int page, Pancreas* pancreas)
{
    pancreas->UpdateParameters(new Params(p0, psc, dmax, gage, page));
}

vector<double> FullSimulation_biphasic(double p0_1, double psc_1, int dmax_1, int gage_1, int page, double p0_2, double psc_2, int dmax_2, int gage_2, double tau, double startVolume, int simtime)
{
	
	Params* parameters = new Params(p0_1, psc_1, dmax_1, gage_1, page);
	vector<Cell*> empty;
	Pancreas* pancreas = new Pancreas(empty, parameters);
	// start with just one infected cancer cell nearest to (0, 0)
	pancreas->CreateInitialTumour();

	// pre-observation phase - run until tumour reaches start volume
	double volume = 0;
	int days1 = 0;
	while (volume < startVolume && days1 < 200)
		volume = pancreas->SimulateOneDay(days1++);
	
	
	// observation phase 1
	vector<double> Tvolume(simtime);
	
	int days2 = 0;
	while (days2 < tau)
	{
		Tvolume[days2] = pancreas->SimulateOneDay(1);
		days2++;
	}

	pancreas->UpdateParameters(new Params(p0_2, psc_2, dmax_2, gage_2, page));
	
	// observation phase 2
	while (days2 < simtime)
	{
		Tvolume[days2] = pancreas->SimulateOneDay(1);
		days2++;
	}				

	// delete stuff?
	delete (parameters);
	delete (pancreas);	
	
	return Tvolume;
}

vector<double> FullSimulation(double p0, double psc, int dmax, int gage, int page, double startVolume, int simtime)
{
	
	Params* parameters = new Params(p0, psc, dmax, gage, page);
	vector<Cell*> empty;
	Pancreas* pancreas = new Pancreas(empty, parameters);
	// start with just one infected cancer cell nearest to (0, 0)
	pancreas->CreateInitialTumour();

	// pre-observation phase - run until tumour reaches start volume
	double volume = 0;
	int days1 = 0;
	while (volume < startVolume && days1 < 200)
		volume = pancreas->SimulateOneDay(days1++);
	
	
	// pre-observation phase - run until tumour reaches start volume
	vector<double> Tvolume(simtime);
	int days2 = 0;
	while (days2 < simtime)
	{
		Tvolume[days2] = pancreas->SimulateOneDay(1);
		days2++;
	}

	// delete stuff?
	delete (parameters);
	delete (pancreas);	
	
	return Tvolume;
}